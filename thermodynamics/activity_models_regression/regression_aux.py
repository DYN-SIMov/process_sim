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
from optimization import PymooCallbackHandler, Callback
# from optimization import DIPPR_polynomial_regression_GA
from data_handling import VLEData

from optimization import PymooPolynomialEstimator


class BinaryInteractionParametersRegression(): 

    def __init__(self,
                 activity_model_regression: ActivityModelRegressionInterface = None,
                 equation_of_state: EquationOfStateInterface = None,
                 VLE_data: VLEData = None,
                 polynomial: PolynomialExponentialDIPPR = None) -> None: 

        self.VLE_data = VLE_data
        self.polynomial = polynomial
        
        pure_component_data_backend = PureComponentDataBackend(components=self.VLE_data.components)
        self.eos_backend = equation_of_state(components=self.VLE_data.components,
                                             pure_component_data_backend=pure_component_data_backend)
        self.activity_model_backend = activity_model_regression(components=self.VLE_data.components,
                                                                pure_component_data_backend=pure_component_data_backend)
        
        self.elementwise_opt_results: list = None
        self.BIP_polynomial_coeffs: dict = None

        pass
        
     

    def regress_BIP_parameters_elementwise(self) -> None:
        " Function to regress Wilson BIP parameters for each temperature point (i.e., data_set) individually. "

        opt_data_results = []

        for T_x_y_point in self.VLE_data.T_x_y_points:

            regression_params = {'y1': [point.y1_mol_frac for point in T_x_y_point.data],
                                 'x1': [point.x1_mol_frac for point in T_x_y_point.data],
                                 'saturation_pressure_Pa_1': T_x_y_point.comp_1_saturation_pressure_Pa,
                                 'saturation_pressure_Pa_2': T_x_y_point.comp_2_saturation_pressure_Pa,
                                 'pressure_Pa': [point.pressure_Pa for point in T_x_y_point.data],
                                 'temperature_K': T_x_y_point.temperature_K,
                                 'eos_backend': self.eos_backend if self.eos_backend is not None else None}

            result = sp_minimize(fun = self.activity_model_backend.objective_function_elementwise,
                                 x0 = self.activity_model_backend.initial_guess_elementwise(initial_guess=opt_data_results[-1] 
                                                                                            if len(opt_data_results) > 0 else None),
                                 args = (regression_params,),
                                 method = 'SLSQP',
                                 bounds=self.activity_model_backend.get_bounds_elementwise()) 

            msg = self.activity_model_backend.get_message_elementwise(result=result,
                                                                      regression_params=regression_params,
                                                                      components=self.VLE_data.components)

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
        self.BIP_polynomial_coeffs = BIP_coeffs
        
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
                local_optimizer_maxiter=int(1e4)
            )
        else: 
            memetic_callback = Callback()

        problem = PymooPolynomialEstimator(
            number_of_parameters=len(lower_bounds),
            n_jobs=n_jobs,
            objective_function=activity_model.objective_function_from_VLE,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            VLE_data=self.VLE_data,
            polynomial=self.polynomial,
            eos_backend=self.eos_backend
        )

        algorithm = GA(pop_size=500,
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
        self.BIP_polynomial_coeffs = BIP_coeffs


        pass


    """
    
    def _objective_function_for_DIPPR_4p_polynomial_from_VLE(self,
                                                             x:np.ndarray) -> float: 

        A1, B1, C1, D1 = x[:4]
        A2, B2, C2, D2 = x[4:]
        
        temperature_K_data = []
        pressure_Pa_data = []
        saturation_pressure_Pa_1_data = []
        saturation_pressure_Pa_2_data = []
        x1_exp_data = [] 
        y1_exp_data = [] 
        
        objective_function_values = []

        for data_set in self.data_set:

            x1_exp_data.extend(data_set['x_data'])
            y1_exp_data.extend(data_set['y_data'])
            temperature_K_data.extend([data_set['temperature_K']] * len(data_set['x_data']))
            pressure_Pa_data.extend(data_set['pressure_Pa'])
            saturation_pressure_Pa_1_data.extend([data_set['saturation_pressure_Pa_1']] * len(data_set['x_data']))
            saturation_pressure_Pa_2_data.extend([data_set['saturation_pressure_Pa_2']] * len(data_set['x_data']))
            
            pass

        test = self.get_all_data_points()
            

        for k in range(0,len(temperature_K_data)): 

            temperature_K = temperature_K_data[k]
            pressure_Pa = pressure_Pa_data[k]
            x1_exp = x1_exp_data[k]
            x2_exp = 1.0 - x1_exp
            y1_exp = y1_exp_data[k]
            y2_exp = 1.0 - y1_exp
            saturation_pressure_Pa_1 = saturation_pressure_Pa_1_data[k]
            saturation_pressure_Pa_2 = saturation_pressure_Pa_2_data[k]


            lambda_12 = self._DIPPR_4p_polynomial_Wilson(coeffs = [A1, B1, C1, D1],
                                                         temperature_K = temperature_K)
            lambda_21 = self._DIPPR_4p_polynomial_Wilson(coeffs = [A2, B2, C2, D2],
                                                         temperature_K = temperature_K)
            
            gamma_1_calc, gamma_2_calc = self.activity_model_backend.get_activity_coefs(lambda_12 = lambda_12,
                                                                                        lambda_21 = lambda_21,
                                                                                        x_val = x1_exp)
            
            if self.equation_of_state is not None:
                fugacity_coef_1, fugacity_coef_2  = self.eos_backend.get_fugacity_coefs(temperature_K = temperature_K,
                                                                                        pressure_Pa = pressure_Pa,
                                                                                        molar_composition = np.array([y1_exp, y2_exp])) 
            else: 
                fugacity_coef_1, fugacity_coef_2  = np.array([1.0, 1.0])  

            y1_calc = x1_exp * gamma_1_calc * saturation_pressure_Pa_1 / (pressure_Pa * fugacity_coef_1)
            y2_calc = x2_exp * gamma_2_calc * saturation_pressure_Pa_2 / (pressure_Pa * fugacity_coef_2)

            weight = np.abs(0.5 - x1_exp) * 10

            objective_function_values.append(
                ((y1_calc - y1_exp) / y1_exp)**2 * weight + 
                ((y2_calc - y2_exp) / y2_exp)**2 * weight
            )
     
        return sum(objective_function_values) 

    """

    """
    def estimate_DIPPR_polynomial_from_VLE_data(self): 



        # NOTE: continue from here

        problem = DIPPR_polynomial_regression_GA(objective_function=self._objective_function_for_DIPPR_4p_polynomial_from_VLE,
                                                 number_of_parameters=8,
                                                 lower_bounds=[-1e3,-1e3,-1e3,-1e0,-1e3,-1e3,-1e3,-1e0],
                                                 upper_bounds=[ 1e3, 1e3, 1e3, 1e0, 1e3, 1e3, 1e3, 1e0],
                                                 n_jobs=1)

        init_pop_single = np.random.uniform(low=[-1e3,-1e3,-1e3,-1e0],
                                            high=[ 1e3, 1e3, 1e3, 1e0],
                                            size=(500,4))
        init_pop_total = []

        for k in range(len(init_pop_single)): 
            init_pop_total.append(np.insert(arr=init_pop_single[k],
                                            obj=0,
                                            values=init_pop_single[k]))
            
        init_pop_pymoo = Population.new("X", init_pop_total)


        algorithm = GA(pop_size=500,
                       eliminate_duplicates=True,
                       sampling=init_pop_pymoo)
        
        results = pymoo_minimize(problem,
                             algorithm,
                             termination=("n_gen", 2000),
                             seed=1,
                             verbose=True,
        )
        
        if results.message is None:
            self.DIPPR_polynomial_coeffs = {
                'Lambda_12': results.X[:4],
                'Lambda_21': results.X[4:]
            }

            msg = (f"\n DIPPR 4th order polynomial regression of Wilson BIP parameters converged successfully. \n"
                    f" Fitted coefficients for Lambda_12: A = {results.X[0]:.4f}, B = {results.X[1]:.4f}, "
                    f"C = {results.X[2]:.4f}, D = {results.X[3]:.4f}."
                    f" Residual = {results.F[0]:.4e}. "
                    f"\n Fitted coefficients for Lambda_21: A = {results.X[4]:.4f}, B = {results.X[5]:.4f}, "
                    f"C = {results.X[6]:.4f}, D = {results.X[7]:.4f}. "
                    f" Residual = {results.F[0]:.4e}. ")
            print(colored(msg, 'green'))
        else: 
            raise RuntimeError(" DIPPR polynomial regression of Wilson BIP parameters did not converge. ")



        pass
        
    """




    def results_visualization(self,
                              get_parity_plot,
                              get_VLE_curve) -> None:
        
        temperature_K_data = []
        pressure_Pa_data = []
        x1_exp_data = []
        y1_exp_data = []
        saturation_pressure_Pa_1_data = []
        saturation_pressure_Pa_2_data = []

        for T_x_y_point in self.VLE_data.T_x_y_points:
            x1_exp_data.extend([datum.x1_mol_frac for datum in T_x_y_point.data])
            y1_exp_data.extend([datum.y1_mol_frac for datum in T_x_y_point.data])
            pressure_Pa_data.extend([datum.pressure_Pa for datum in T_x_y_point.data])
            temperature_K_data.extend([T_x_y_point.temperature_K] * len(T_x_y_point.data))
            saturation_pressure_Pa_1_data.extend([T_x_y_point.comp_1_saturation_pressure_Pa] * len(T_x_y_point.data))
            saturation_pressure_Pa_2_data.extend([T_x_y_point.comp_2_saturation_pressure_Pa] * len(T_x_y_point.data))

        x1_exp_data_plot        = []
        y1_exp_data_plot        = []
        x2_exp_data_plot        = []
        y2_exp_data_plot        = []
        y1_calc_data_plot       = []
        y2_calc_data_plot       = []
        temperature_K_data_plot = []

        for k in range(len(x1_exp_data)):

            temperature_K = temperature_K_data[k]
            pressure_Pa = pressure_Pa_data[k]
            x1_exp = x1_exp_data[k] 
            y1_exp = y1_exp_data[k]
            saturation_pressure_Pa_1 = saturation_pressure_Pa_1_data[k]
            saturation_pressure_Pa_2 = saturation_pressure_Pa_2_data[k]

            BIP_values = []
            for k in range(0, len(self.BIP_polynomial_coeffs)):
                BIP_val = self.polynomial.evaluate(temperature_K = temperature_K,
                                                   coeffs = self.BIP_polynomial_coeffs[k])
                BIP_values.append(BIP_val)

            gamma_1_calc, gamma_2_calc = self.activity_model_backend.get_activity_coefs(theta=BIP_values,
                                                                                        x_val=x1_exp)

            if self.eos_backend is not None:
                fugacity_coef_1, fugacity_coef_2  = self.eos_backend.get_fugacity_coefs(temperature_K = temperature_K,
                                                                                            pressure_Pa = pressure_Pa,
                                                                                            molar_composition = np.array([y1_exp, 1 - y1_exp])) 
            else: 
                fugacity_coef_1, fugacity_coef_2  = np.array([1.0, 1.0])   # ideal gas assumption if no EoS backend is specified


            y1_calc = x1_exp * gamma_1_calc * saturation_pressure_Pa_1 / (pressure_Pa * fugacity_coef_1)
            y2_calc = (1 - x1_exp) * gamma_2_calc * saturation_pressure_Pa_2 / (pressure_Pa * fugacity_coef_2)

            x1_exp_data_plot.append(x1_exp)
            y1_exp_data_plot.append(y1_exp)
            x2_exp_data_plot.append(x1_exp)
            y2_exp_data_plot.append(y1_exp)
            y1_calc_data_plot.append(y1_calc)
            y2_calc_data_plot.append(y2_calc)

        # Estimating R2 coefficients  
        slope_y1, intercept_y1, r_value_y1, p_value_y1, std_err_y1 = linregress(y1_exp_data_plot, y1_calc_data_plot)

        if get_parity_plot is True: 
            plt.figure(figsize=(6,6))
            plt.scatter(y1_exp_data_plot, y1_calc_data_plot, color='red', label=f'y_{self.VLE_data.components[0]}')
            plt.plot([0, 1], [0, 1], 'k--', label='reference line')
            plt.xlabel(f'Experimental y_{self.VLE_data.components[0]}')
            plt.ylabel(f'Calculated y_{self.VLE_data.components[0]}')
            plt.title(f'Parity Plot for VLE of {self.VLE_data.components[0]} and {self.VLE_data.components[1]} \n'
                      f'R2 {self.VLE_data.components[0]} =  {r_value_y1**2:.3f}')   
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.legend()
            plt.show(block=False)

        if get_VLE_curve is True: 
            plt.figure(figsize=(6,6))
            plt.scatter(x1_exp_data_plot, y1_exp_data_plot, color='red', label=f'exp data for {self.VLE_data.components[0]}')
            plt.plot(x1_exp_data_plot, y1_calc_data_plot, color='blue', label=f'calc data for {self.VLE_data.components[0]}')
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


