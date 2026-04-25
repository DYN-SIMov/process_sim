import numpy as np
import matplotlib.pyplot as plt 

from scipy.optimize import minimize as sp_minimize        # for regression 
from scipy.stats import linregress                        # for R2 calculation
from termcolor import colored                             # for colored text output

# for regression of DIPPR parameters directly from VLE data
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import EquationOfStateInterface 
from thermodynamics.core.properties import PureComponentDataBackend

from activity_models_aux import ActivityModelRegressionInterface
from optimization import PolynomialExponentialDIPPR, PolynomialElementwiseEstimator, PolynomialInterface
from optimization import PolynominalFormInterface, AbsoluteForm, NormalizedForm
from optimization import PymooCallbackHandler, Callback
# from optimization import DIPPR_polynomial_regression_GA
from data_handling import VLEData

from optimization import PymooPolynomialEstimator


class BinaryInteractionParametersRegression(): 

    def __init__(self,
                 activity_model_regression: ActivityModelRegressionInterface = None,
                 equation_of_state: EquationOfStateInterface = None,
                 VLE_data: VLEData = None,
                 polynomial: PolynomialExponentialDIPPR = None,
                 polynomial_form: PolynominalFormInterface = AbsoluteForm) -> None: 

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
        
        self.elementwise_opt_results: list = None
        self.BIP_polynomial_coeffs: dict = None

        pass
        
     

    def regress_BIP_parameters_elementwise(self) -> None:
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

            msg = self.activity_model_backend.get_message_elementwise(
                result=result,
                regression_params=regression_params,
                components=self.VLE_data.components
            )

            opt_data_results.append(result.x)

        self.elementwise_opt_results = np.array(opt_data_results)

        pass



    def estimate_polynomial_from_elementwise_optimisation(self) -> None:

        " Method for regressing DIPPR polynomial parameters based on results of elementwise "
        " regression of Wilson Binary Interaction Parameters"
        
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
            estimation_results=estimation_results
        )

        BIP_coeffs = [estimation_results[k].x for k in range(len(estimation_results))]
        self.BIP_polynomial_coeffs = [
            self.polynomial.get_absolute_coeffs(
                coeffs=BIP_coeffs[k]
            ) for k in range(len(BIP_coeffs))
        ]
        
        pass



    def estimate_polynomial_from_VLE_data(self,
                                          n_jobs:int = 1,
                                          is_memetic:bool = True,
                                          verbose:bool = False) -> None: 

        " Method for regressing parameters directly from VLE data "
        activity_model = self.activity_model_backend

        scipy_bounds = self.polynomial.get_bounds_scipy()
        lower_bounds = [bound[0] for bound in scipy_bounds]*activity_model.number_of_BIP_parameters
        upper_bounds = [bound[1] for bound in scipy_bounds]*activity_model.number_of_BIP_parameters

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
            number_of_parameters=len(lower_bounds),
            n_jobs=n_jobs,
            objective_function=self._objective_function_from_VLE,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            VLE_data=self.VLE_data,
            polynomial=self.polynomial,
        )

        algorithm = GA(pop_size=1000,
                       eliminate_duplicates=True)
        
        results = pymoo_minimize(
            problem=problem,
            algorithm=algorithm,
            termination=("n_gen", 500),
            seed=1,
            verbose=verbose,
            callback=memetic_callback
        )

        activity_model.get_message_estimation_from_VLE(
            coeffs = results.X,
            total_residual = results.F[0]
        )

        BIP_coeffs = results.X.reshape((activity_model.number_of_BIP_parameters, 
                                        self.polynomial.degree))
        self.BIP_polynomial_coeffs = [
            self.polynomial.get_absolute_coeffs(
                coeffs=BIP_coeffs[k]
            ) for k in range(len(BIP_coeffs))
        ]


        pass


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
            theta = BIP_coeffs,
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
                BIP_coeffs = theta
            )

            y1_calc_val = output['y_calc_val_1']
            y2_calc_val = output['y_calc_val_2']
            pressure_total_Pa_calc = output['pressure_total_Pa_calc']
    
            error = (
                (y1_exp_val - y1_calc_val)**2 + 
                (y2_exp_val - y2_calc_val)**2 + 
                ((pressure_Pa_data[k] - pressure_total_Pa_calc)/pressure_Pa_data[k])**2
            )
            error_data.append(error)

        return sum(error_data)
    

    def _objective_function_from_VLE(self,
                                     coeffs: np.ndarray,
                                     polynomial,
                                     VLE_data) -> float:

        param_coeffs = coeffs.reshape(
            (self.activity_model_backend.number_of_BIP_parameters, polynomial.degree)
        )
        
        error_data = []
        for T_x_y_point in VLE_data.T_x_y_points:
            theta = []

            BIP_12_calc = polynomial.evaluate(temperature_K = T_x_y_point.temperature_K,
                                            coeffs = param_coeffs[0])
            BIP_21_calc = polynomial.evaluate(temperature_K = T_x_y_point.temperature_K,
                                            coeffs = param_coeffs[1])
            theta = np.array([BIP_12_calc, BIP_21_calc])

            error_val = self._objective_function_elementwise(
                theta = theta,
                regression_params = {
                    'x1': np.array([point.x1_mol_frac for point in T_x_y_point.data]),
                    'y1': np.array([point.y1_mol_frac for point in T_x_y_point.data]),
                    'pressure_Pa': np.array([point.pressure_Pa for point in T_x_y_point.data]),
                    'temperature_K': T_x_y_point.temperature_K,
                    'saturation_pressure_Pa_1': T_x_y_point.comp_1_saturation_pressure_Pa,
                    'saturation_pressure_Pa_2': T_x_y_point.comp_2_saturation_pressure_Pa,
                })
            error_data.append(error_val)

        return sum(error_data)


    def results_visualization(self,
                              get_parity_plot,
                              get_VLE_curve) -> None:
        
        visualization_data = self._extract_visualization_data()
        y1_exp_data_plot = visualization_data['y1_exp_data']
        y1_calc_data_plot = visualization_data['y1_calc_data']
        x1_exp_data_plot = visualization_data['x1_exp_data']
        y1_calc_elementwise_data_plot = visualization_data['y1_calc_elementwise_data']
        y2_calc_elementwise_data_plot = visualization_data['y2_calc_elementwise_data']

        # Estimating R2 coefficients  
        [slope_y1, 
         intercept_y1, 
         r_value_y1, 
         p_value_y1, 
         std_err_y1] = linregress(y1_exp_data_plot, y1_calc_data_plot)

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
                      f'R2 {self.VLE_data.components[0]} =  {r_value_y1**2:.3f}')   
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.show(block=False)

        if get_VLE_curve is True: 
            plt.figure(figsize=(6,6))
            plt.scatter(x1_exp_data_plot, y1_exp_data_plot, 
                        color='red', label=f'exp data for {self.VLE_data.components[0]}'
            )
            plt.plot(x1_exp_data_plot, y1_calc_data_plot, 
                     color='blue', label=f'calc data for {self.VLE_data.components[0]}'
            )
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
        
        input("Press Enter to continue...")

        pass


    def _extract_visualization_data(self) -> dict: 

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
            for j in range(0, len(self.BIP_polynomial_coeffs)):
                BIP_val = self.polynomial.evaluate(temperature_K = temperature_K,
                                                   coeffs = self.BIP_polynomial_coeffs[j])
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

            if self.elementwise_opt_results is not None:
                pt_idx = point_indices[k]
                BIP_elementwise_values = self.elementwise_opt_results[pt_idx]
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
            'y2_calc_elementwise_data': y2_calc_elementwise_data_plot
        }




    