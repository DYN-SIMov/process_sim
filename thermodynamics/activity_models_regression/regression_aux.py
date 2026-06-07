import numpy as np
import matplotlib.pyplot as plt 

from scipy.optimize import minimize as sp_minimize        # for regression 
from scipy.stats import linregress                        # for R2 calculation
from termcolor import colored                             # for colored text output

# for regression of DIPPR parameters directly from VLE data
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import GA, FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import EquationOfStateInterface 
from thermodynamics.core.properties import PureComponentDataBackend

from activity_models_aux import ActivityModelRegressionInterface
from activity_models_aux import BinaryInteractionParameter
from optimization import PolynomialExponentialDIPPR, PolynomialElementwiseEstimator, PolynomialInterface
from optimization import PolynominalFormInterface, AbsoluteForm, NormalizedForm
from optimization import PymooCallbackHandler, Callback
# from optimization import DIPPR_polynomial_regression_GA
from data_handling import VLEData

from optimization import PymooPolynomialEstimator
from optimization import OptimizationVectorMapper
from optimization import OwnBiasedSampling

from typing import Type
from enum import Enum

from dataclasses import replace as replace_dataclass_field



class RegressionMethod(Enum):
    ELEMENTWISE = 'Elementwise regression of each T-x-y point'
    DIRECT_VLE = 'Direct regression from VLE data using memetic algorithm'



class BinaryInteractionParametersRegression(): 

    def __init__(self,
                 activity_model_regression: Type[ActivityModelRegressionInterface] = None,
                 equation_of_state: Type[EquationOfStateInterface] = None,
                 VLE_data: VLEData = None,
                 polynomial: PolynomialExponentialDIPPR = None,
                 polynomial_form: Type[PolynominalFormInterface] = AbsoluteForm) -> None: 

        self.VLE_data = VLE_data
        self.polynomial = polynomial_form(polynomial) if polynomial is not None else None

        pure_component_data_backend = PureComponentDataBackend(
            components=self.VLE_data.components
        )

        self.eos_backend = equation_of_state(
            components=self.VLE_data.components,
            pure_component_data_backend=pure_component_data_backend
        ) if equation_of_state is not None else None

        self.activity_model_backend = activity_model_regression(
            components=self.VLE_data.components,
            pure_component_data_backend=pure_component_data_backend
        )

        self.parameter_mapper = OptimizationVectorMapper(
            activity_model=self.activity_model_backend,
            polynomial=self.polynomial
        )

        self.elementwise_opt_results: list = None
        self.regression_results_cache: dict = {}
        self.latest_regression_results: list[BinaryInteractionParameter] = None

        pass
        

    def estimate_polynomial_elementwise(self) -> None:

        " Method for regressing BIP parameters for each T-x-y point individually and then "
        " regressing the polynomial coefficients based on the results of elementwise regression. "
        " The elementwise regression does not consider potential variation in temperature dependant "
        " BIPs - they are considered as constants (based on the initial guess value provided)" 

        self._regress_BIP_parameters_elementwise()
        self._estimate_polynomial_from_elementwise_optimisation()

        pass


    def _regress_BIP_parameters_elementwise(self,
                                            verbose: bool = True) -> None:
        " Function to regress Wilson BIP parameters for each temperature point "
        " (i.e., data_set) individually. "

        opt_data_results = []

        for T_x_y_point in self.VLE_data.T_x_y_points:

            regression_params = {
                'y1': [point.y1_mol_frac for point in T_x_y_point.data],
                'x1': [point.x1_mol_frac for point in T_x_y_point.data],
                'saturation_pressure_Pa_1': T_x_y_point.comp_1_saturation_pressure_Pa,
                'saturation_pressure_Pa_2': T_x_y_point.comp_2_saturation_pressure_Pa,
                'pressure_Pa': [point.pressure_Pa for point in T_x_y_point.data],
                'temperature_K': T_x_y_point.temperature_K
            }

            result = sp_minimize(
                fun = self._objective_function_elementwise,
                x0 = self.activity_model_backend.initial_guess_elementwise(
                    initial_guess=opt_data_results[-1] if len(opt_data_results) > 0 else None
                ),
                args = (regression_params,),
                method = 'SLSQP',
                bounds=self.activity_model_backend.get_bounds_elementwise()
            ) 

            if verbose:
                msg = self.activity_model_backend.get_message_elementwise(
                    result=result,
                    regression_params=regression_params,
                    components=self.VLE_data.components
                )

            opt_data_results.append(result.x)

        self.elementwise_opt_results = np.array(opt_data_results)

        pass


    def _estimate_polynomial_from_elementwise_optimisation(self) -> None:

        " Method for regressing DIPPR polynomial parameters based on results of elementwise "
        " regression of Binary Interaction Parameters"
        
        if self.elementwise_opt_results is None:
            raise ValueError(" Elementwise optimisation results not found. "
                             " Run regress_BIP_parameters_elementwise() first. ")

        temperature_K_data = [T_x_y_point.temperature_K for T_x_y_point in self.VLE_data.T_x_y_points]
        BIP_elementwise_results = self.elementwise_opt_results

        elementwise_estimator = PolynomialElementwiseEstimator(
            temperature_K_data = temperature_K_data,
            BIP_elementwise_results = BIP_elementwise_results,
            polynomial = self.polynomial
        )
        estimation_results = elementwise_estimator.estimate_coefficients()
        self.activity_model_backend.get_polynomial_coeffs_estimation_message(
            estimation_results=estimation_results,
            polynomial=self.polynomial
        )

        # NOTE: elementwise regression considers only temperature dependant BIPs
        bip_names = [bip.name for bip in self.activity_model_backend.BIPs
                      if bip.is_regressed and bip.is_temperature_dependant]
        BIP_estimation_results = {
            bip_names[k]: estimation_results[k].x
            for k in range(len(estimation_results))
        }

        self._record_regression_results(
            regression_method=RegressionMethod.ELEMENTWISE,
            BIP_estimation_results=BIP_estimation_results
        )

        pass



    def estimate_polynomial_from_VLE_data(self,
                                          n_jobs:int = 1,
                                          is_memetic:bool = True,
                                          verbose:bool = False,
                                          use_biased_initialization: bool = True) -> None: 

        " Method for regressing parameters directly from VLE data "
        activity_model = self.activity_model_backend

        if is_memetic: 
            memetic_callback = PymooCallbackHandler(
                n_gens_skipped=20, 
                verbose=verbose,
                local_optimizer_maxiter=int(1e4),
                n_candidates_selected=4, 
            )
        else: 
            memetic_callback = Callback()

        problem = PymooPolynomialEstimator(
            n_jobs=n_jobs,
            objective_function=self._objective_function_from_VLE,
            VLE_data=self.VLE_data,
            polynomial=self.polynomial,
            parameter_mapper=self.parameter_mapper
        )

        if use_biased_initialization is True: 
            if self.regression_results_cache[RegressionMethod.ELEMENTWISE] is None:
                msg = (" No elementwise optimization results found to implement biased "
                       "initialization. Running elementwise optimization now. ")
                print(msg)
                self.estimate_polynomial_elementwise()

            BIP_config = self.regression_results_cache[
                RegressionMethod.ELEMENTWISE
            ]['regressed_BIPs']

            initial_population = OwnBiasedSampling(
                BIP_config = BIP_config,
                fraction_of_biased_samples=0.2,
                bias_strength=0.5
            )
        else: 
            initial_population = FloatRandomSampling()

        algorithm = GA(pop_size=1000,
                       eliminate_duplicates=True,
                       sampling=initial_population)
        
        results = pymoo_minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", 100),
            seed=1,
            verbose=verbose,
            callback=memetic_callback
        )

        BIP_data_vect = self.parameter_mapper.decode_pymoo_vector(
            optimization_vector=results.X
        )

        activity_model.get_message_estimation_from_VLE(
            decoded_bip_data=BIP_data_vect,
            polynomial=self.polynomial,
            total_residual=results.F[0]
        )
        
        bip_names = [bip.name for bip in self.activity_model_backend.BIPs
                      if bip.is_regressed]
        
        BIP_estimation_results = {
            bip_names[k]: BIP_data_vect[bip_names[k]]
            for k in range(len(BIP_data_vect))
        }

        self._record_regression_results(
            regression_method=RegressionMethod.DIRECT_VLE,
            BIP_estimation_results=BIP_estimation_results
        )

        pass


    def compare_regression_methods(self) -> None:
        
        for method, results in self.regression_results_cache.items():
            print(f" Regression method: {method.value} ")
            for bip in results['regressed_BIPs']:
                if hasattr(bip.value, 'shape'):
                    print(f"   {bip.name}:"
                          f" {np.array2string(
                              bip.value, 
                              formatter={'float_kind': lambda x: f'{x:6.3f}'})}"
                    )
                else:
                    print(f"   {bip.name}: {bip.value:.3f}")
            print(f" Goodness of fit (R^2): {results['goodness_of_fit']:.4f}")
            print("-"*50)
        print("\n"*2)


    def _estimate_y_calculated(self,
                               x1_val:float,
                               y1_val:float,
                               pressure_Pa:float,
                               temperature_K:float,
                               saturation_pressure_Pa_1:float,
                               saturation_pressure_Pa_2:float,
                               BIP_coeffs:np.ndarray) -> dict: 

        x2_val = 1 - x1_val
        y2_val = 1 - y1_val
        
        gamma_1_calc, gamma_2_calc = self.activity_model_backend.get_activity_coefs(
            BIP_coeffs = BIP_coeffs,
            x_val = x1_val
        )

        if self.eos_backend is not None:
            fugacity_coef_1, fugacity_coef_2  = self.eos_backend.get_fugacity_coefs(
                temperature_K = temperature_K,
                pressure_Pa = pressure_Pa,
                molar_composition = np.array([y1_val, y2_val])) 
        else: 
            # ideal gas assumption if no EoS backend is specified
            fugacity_coef_1, fugacity_coef_2  = np.array([1.0, 1.0])   

        # based on the modified Raoult's law: y_i * fi_i * P_total = x_i * gamma_i * P_sat_i 
        pressure_total_Pa_calc = (
            (x1_val * gamma_1_calc * saturation_pressure_Pa_1) / fugacity_coef_1 + 
            (x2_val * gamma_2_calc * saturation_pressure_Pa_2) / fugacity_coef_2
        )
        
        y_calc_val_1 = (
            x1_val * gamma_1_calc * saturation_pressure_Pa_1 / 
            (pressure_total_Pa_calc * fugacity_coef_1)
        )
        y_calc_val_2 = (
            x2_val * gamma_2_calc * saturation_pressure_Pa_2 / 
            (pressure_total_Pa_calc * fugacity_coef_2)
        )
        
        return {
            'y_calc_val_1': y_calc_val_1,
            'y_calc_val_2': y_calc_val_2,
            'pressure_total_Pa_calc': pressure_total_Pa_calc
        }


    def _objective_function_elementwise(self, 
                                        theta: np.ndarray,
                                        regression_params) -> float:

        """
        Objective function for elementwise estimation of BIP parameters based on 
        the modified Raoult's law: y_i * fi_i * P_total= x_i * gamma_i * P_sat_i
        """

        x_exp_data = regression_params['x1']
        y_exp_data = regression_params['y1']
        pressure_Pa_data = regression_params['pressure_Pa']
        temperature_K = regression_params['temperature_K']
        saturation_pressure_Pa_1 = regression_params['saturation_pressure_Pa_1']
        saturation_pressure_Pa_2 = regression_params['saturation_pressure_Pa_2']

        BIP_coeffs = self.parameter_mapper.elementwise_regression_adapter(
            optimization_vector=theta
        )

        error_data = []
        for k in range(len(x_exp_data)):

            x1_exp_val = x_exp_data[k]
            y1_exp_val = y_exp_data[k]
            x2_exp_val = 1 - x1_exp_val
            y2_exp_val = 1 - y1_exp_val

            output = self._estimate_y_calculated(
                x1_val = x1_exp_val,
                y1_val = y1_exp_val,
                pressure_Pa = pressure_Pa_data[k],
                temperature_K = temperature_K,
                saturation_pressure_Pa_1 = saturation_pressure_Pa_1,
                saturation_pressure_Pa_2 = saturation_pressure_Pa_2,
                BIP_coeffs = BIP_coeffs
            )

            y1_calc_val = output['y_calc_val_1']
            y2_calc_val = output['y_calc_val_2']
            pressure_total_Pa_calc = output['pressure_total_Pa_calc']
    
            error = (
                ((y1_exp_val - y1_calc_val)/y1_exp_val)**2 + 
                ((y2_exp_val - y2_calc_val)/y2_exp_val)**2 + 
                ((pressure_Pa_data[k] - pressure_total_Pa_calc)/pressure_Pa_data[k])**2
            )
            error_data.append(error)

        return sum(error_data)
    

    def _objective_function_from_VLE(self,
                                     coeffs: np.ndarray,
                                     polynomial,
                                     VLE_data) -> float:

        decoded_params = self.parameter_mapper.decode_pymoo_vector(
            optimization_vector=coeffs
        )

        error_data = []
        for T_x_y_point in VLE_data.T_x_y_points:

            BIP_estimates = []
            for bip in self.activity_model_backend.BIPs:
                if not bip.is_regressed:
                    continue
                if bip.is_temperature_dependant:
                    BIP_estimates.append(polynomial.evaluate(
                        temperature_K=T_x_y_point.temperature_K,
                        coeffs=decoded_params[bip.name]
                    ))
                else:
                    BIP_estimates.append(decoded_params[bip.name])

            error_val = self._objective_function_elementwise(
                theta=BIP_estimates,
                regression_params={
                    'x1': [point.x1_mol_frac for point in T_x_y_point.data],
                    'y1': [point.y1_mol_frac for point in T_x_y_point.data],
                    'pressure_Pa': [point.pressure_Pa for point in T_x_y_point.data],
                    'temperature_K': T_x_y_point.temperature_K,
                    'saturation_pressure_Pa_1': T_x_y_point.comp_1_saturation_pressure_Pa,
                    'saturation_pressure_Pa_2': T_x_y_point.comp_2_saturation_pressure_Pa,
                })
            error_data.append(error_val)

        return sum(error_data)


    def _calculate_goodness_of_fit(self,
                                   regressed_BIPs: list[BinaryInteractionParameter]) -> float:
        """
        Calculates the goodness of fit (R^2) for the regressed model.
        """
        visualization_data = self._extract_visualization_data(regressed_BIPs=regressed_BIPs)
        y1_exp_data_plot = visualization_data['y1_exp_data']
        y1_calc_data_plot = visualization_data['y1_calc_data']
        
        [slope_y1, 
         intercept_y1, 
         r_value_y1, 
         p_value_y1, 
         std_err_y1] = linregress(y1_exp_data_plot, y1_calc_data_plot)
        
        return r_value_y1**2
    

    def _record_regression_results(self, 
                                   BIP_estimation_results: dict, 
                                   regression_method: RegressionMethod) -> None :


        regressed_BIPs = self._get_updated_BIPs(BIP_estimation_results=BIP_estimation_results)
        goodness_of_fit = self._calculate_goodness_of_fit(regressed_BIPs=regressed_BIPs)

        self.regression_results_cache[regression_method] = {
            'regressed_BIPs': regressed_BIPs,
            'goodness_of_fit': goodness_of_fit
        }

        self.latest_regression_results = regressed_BIPs

        pass 

    
    def _get_updated_BIPs(self, 
                          BIP_estimation_results: dict) -> list[BinaryInteractionParameter]:

        updated_BIPs = []
        
        for bip in self.activity_model_backend.BIPs:
            if bip.name in BIP_estimation_results:
                updated_BIP = replace_dataclass_field(
                    bip, 
                    value=BIP_estimation_results[bip.name]
                )
                updated_BIPs.append(updated_BIP)
            else:
                updated_BIP = replace_dataclass_field(
                    bip, 
                    value=bip.initial_guess
                )
                updated_BIPs.append(updated_BIP)

        return updated_BIPs


    def results_visualization(self,
                              get_parity_plot,
                              get_VLE_curve) -> None:
        
        regressed_BIPs = self.latest_regression_results
 
        visualization_data = self._extract_visualization_data(regressed_BIPs=regressed_BIPs)
        y1_exp_data_plot = visualization_data['y1_exp_data']
        y1_calc_data_plot = visualization_data['y1_calc_data']
        x1_exp_data_plot = visualization_data['x1_exp_data']
        y1_calc_elementwise_data_plot = visualization_data['y1_calc_elementwise_data']
        y2_calc_elementwise_data_plot = visualization_data['y2_calc_elementwise_data']
        point_indices = visualization_data['point_indices']

        goodness_of_fit = self._calculate_goodness_of_fit(regressed_BIPs=regressed_BIPs)

        if get_parity_plot is True: 
            plt.figure(figsize=(6,6))
            plt.scatter(y1_exp_data_plot, y1_calc_data_plot, 
                        color='red', label=f'y_{self.VLE_data.components[0]}'
            )
            plt.plot([0, 1], [0, 1], 'k--', label='reference line')
            plt.xlabel(f'Experimental y_{self.VLE_data.components[0]}')
            plt.ylabel(f'Calculated y_{self.VLE_data.components[0]}')
            plt.title(f'Parity Plot for VLE of {self.VLE_data.components[0]} and '
                      f'{self.VLE_data.components[1]} \n'
                      f'R2 {self.VLE_data.components[0]} =  {goodness_of_fit:.3f}')   
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.show(block=False)

        if get_VLE_curve is True: 
            plt.figure(figsize=(6,6))
            plt.scatter(x1_exp_data_plot, y1_exp_data_plot, 
                        color='red', label=f'exp data for {self.VLE_data.components[0]}'
            )
            
            # ----------------------------------------------------------------------------
            # Filtering out datapoints belonging to different T-x-y curves (diffent pressures)
            # ----------------------------------------------------------------------------
            unique_indices = sorted(list(set(point_indices)))
            if len(unique_indices) == len(point_indices):
                x_curve = x1_exp_data_plot
                y_calc_curve = y1_calc_data_plot
                label = f'calc data for {self.VLE_data.components[0]}'
                plt.plot(x_curve, y_calc_curve, color='blue', label=label)
            else: 
                for i, p_idx in enumerate(unique_indices):
                    idx_mask = [k for k, val in enumerate(point_indices) if val == p_idx]
                    x_curve = [x1_exp_data_plot[k] for k in idx_mask]
                    y_calc_curve = [y1_calc_data_plot[k] for k in idx_mask]
                    
                    label = f'calc data for {self.VLE_data.components[0]}' if i == 0 else "_nolegend_"
                    plt.plot(x_curve, y_calc_curve, color='blue', label=label)

            if self.elementwise_opt_results is not None: 
                plt.scatter(x1_exp_data_plot, y1_calc_elementwise_data_plot, 
                            color='green', 
                            marker='x',
                            label=f'elementwise regression results for {self.VLE_data.components[0]}'
                )
            plt.plot([0, 1], [0, 1], 'k--', label='y=x')
            plt.xlabel(f'x_{self.VLE_data.components[0]}')
            plt.ylabel(f'y_{self.VLE_data.components[0]}')
            plt.title(f'VLE curve for {self.VLE_data.components[0]} and {self.VLE_data.components[1]}')   
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.show(block=False)

            pass 
        
        input("Press any key to exit the visualization...")
        plt.close('all')

        pass


    def _extract_visualization_data(self,
                                    regressed_BIPs: list[BinaryInteractionParameter]) -> dict: 

        """
        Method for extracting the data for visualization of regression results.

        Args: 
            None
        Returns:
            dict: A dictionary containing the data for visualization, including:
                - x1_exp_data: A list of experimental x1 mol fraction data points.
                - y1_exp_data: A list of experimental y1 mol fraction data points.
                - x2_exp_data: A list of experimental x2 mol fraction data points.
                - y2_exp_data: A list of experimental y2 mol fraction data points.
                - y1_calc_data: A list of calculated y1 mol fraction data points based on 
                    the regressed BIP polynomial.
                - y2_calc_data: A list of calculated y2 mol fraction data points based on 
                    the regressed BIP polynomial.
                - y1_calc_elementwise_data: A list of calculated y1 mol fraction data points
                    based on the elementwise regression of BIP parameters.
                - y2_calc_elementwise_data: A list of calculated y2 mol fraction data points
                    based on the elementwise regression of BIP parameters.
                - point_indices: A list of indices corresponding to each data point, indicating
                    which temperature point (i.e., T_x_y_point) it belongs to. This is useful for 
                    keeping track when there are several data points for each T-x-y point. 
                    This variable is also used for visualization of VLE curves, 
                    where the data points from the same T-x-y point are connected with a line.
        """

        temperature_K_data = []
        pressure_Pa_data = []
        x1_exp_data = []
        y1_exp_data = []
        saturation_pressure_Pa_1_data = []
        saturation_pressure_Pa_2_data = []
        point_indices = [] 

        for pt_idx, T_x_y_point in enumerate(self.VLE_data.T_x_y_points):
            x1_exp_data.extend([datum.x1_mol_frac for datum in T_x_y_point.data])
            y1_exp_data.extend([datum.y1_mol_frac for datum in T_x_y_point.data])
            pressure_Pa_data.extend([datum.pressure_Pa for datum in T_x_y_point.data])
            temperature_K_data.extend([T_x_y_point.temperature_K] * len(T_x_y_point.data))
            saturation_pressure_Pa_1_data.extend(
                [T_x_y_point.comp_1_saturation_pressure_Pa] * len(T_x_y_point.data)
            )
            saturation_pressure_Pa_2_data.extend(
                [T_x_y_point.comp_2_saturation_pressure_Pa] * len(T_x_y_point.data)
            )
            point_indices.extend([pt_idx] * len(T_x_y_point.data))

        x1_exp_data_plot        = []
        y1_exp_data_plot        = []

        x2_exp_data_plot        = []
        y2_exp_data_plot        = []
        
        y1_calc_data_plot       = []
        y2_calc_data_plot       = []
        
        y1_calc_elementwise_data_plot = []
        y2_calc_elementwise_data_plot = []

        for k in range(len(x1_exp_data)):

            temperature_K = temperature_K_data[k]
            pressure_Pa = pressure_Pa_data[k]
            x1_exp = x1_exp_data[k] 
            y1_exp = y1_exp_data[k]
            saturation_pressure_Pa_1 = saturation_pressure_Pa_1_data[k]
            saturation_pressure_Pa_2 = saturation_pressure_Pa_2_data[k]

            BIP_values = []
            for bip in regressed_BIPs:
                if not bip.is_regressed:
                    continue
                if bip.is_temperature_dependant:
                    BIP_val = self.polynomial.evaluate(temperature_K = temperature_K,
                                                       coeffs = bip.value)
                else:
                    BIP_val = bip.value
                BIP_values.append(BIP_val)

            output = self._estimate_y_calculated(
                x1_val = x1_exp,
                y1_val = y1_exp,
                pressure_Pa = pressure_Pa,
                temperature_K = temperature_K,
                saturation_pressure_Pa_1 = saturation_pressure_Pa_1,
                saturation_pressure_Pa_2 = saturation_pressure_Pa_2,
                BIP_coeffs = BIP_values
            )

            x1_exp_data_plot.append(x1_exp)
            y1_exp_data_plot.append(y1_exp)

            x2_exp_data_plot.append(x1_exp)
            y2_exp_data_plot.append(y1_exp)

            y1_calc_data_plot.append(output['y_calc_val_1'])
            y2_calc_data_plot.append(output['y_calc_val_2'])

            # optional visualization to check the results of the elementwise regression 
            if self.elementwise_opt_results is not None:

                pt_idx = point_indices[k]
                BIP_elementwise_values = list(self.elementwise_opt_results[pt_idx])
                for bip in regressed_BIPs:
                    if bip.is_regressed and not bip.is_temperature_dependant:
                        BIP_elementwise_values.append(bip.value)

                output_elementwise = self._estimate_y_calculated(
                    x1_val = x1_exp,
                    y1_val = y1_exp,
                    pressure_Pa = pressure_Pa,
                    temperature_K = temperature_K,
                    saturation_pressure_Pa_1 = saturation_pressure_Pa_1,
                    saturation_pressure_Pa_2 = saturation_pressure_Pa_2,
                    BIP_coeffs = BIP_elementwise_values
                )
                y1_calc_elementwise_data_plot.append(output_elementwise['y_calc_val_1'])
                y2_calc_elementwise_data_plot.append(output_elementwise['y_calc_val_2'])

                pass

        return {
            'x1_exp_data': x1_exp_data_plot,
            'y1_exp_data': y1_exp_data_plot,
            'x2_exp_data': x2_exp_data_plot,
            'y2_exp_data': y2_exp_data_plot,
            'y1_calc_data': y1_calc_data_plot,
            'y2_calc_data': y2_calc_data_plot,
            'y1_calc_elementwise_data': y1_calc_elementwise_data_plot,
            'y2_calc_elementwise_data': y2_calc_elementwise_data_plot,
            'point_indices': point_indices
        }




    