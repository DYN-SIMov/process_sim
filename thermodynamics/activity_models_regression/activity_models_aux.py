import numpy as np

from typing import Protocol

import sys
import os

from termcolor import colored
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import WilsonActivityModel


class ActivityModelRegressionInterface(Protocol):
    @staticmethod
    def get_activity_coefs(*args, **kwargs) -> np.ndarray:
        pass

    def objective_function_elementwise(self, 
                                       theta: np.ndarray,
                                       regression_params) -> float:
        pass

    @staticmethod
    def initial_guess_elementwise(initial_guess) -> np.ndarray:
        pass

    @staticmethod
    def get_bounds_elementwise() -> tuple:
        pass

    @staticmethod
    def get_message_elementwise(regression_params: dict,
                                components: list[str],
                                result) -> None:
        pass


class WilsonActivityModelRegression(WilsonActivityModel):
    
    def __init__(self,
                 components,
                 pure_component_data_backend):
        super().__init__(components=components,
                         pure_component_data_backend=pure_component_data_backend)
        pass


    @staticmethod
    def get_activity_coefs(lambda_12: float,
                           lambda_21: float,
                           x_val: np.ndarray) -> np.ndarray:

        """
        Method to get activity coeffcients (gamma_1 and gamma_2) based on Wilson equation
        """

        x_1 = x_val
        x_2 = 1 - x_val

        ln_gamma_1 = (
                        -np.log(x_1 + lambda_12 * x_2) + 
                        x_2 * lambda_12 / (x_1 + lambda_12 * x_2) - 
                        x_2 * lambda_21 / (x_2 + lambda_21 * x_1)
        )

        ln_gamma_2 = (
                        -np.log(x_2 + lambda_21 * x_1) -
                        x_1 * lambda_12 / (x_1 + lambda_12 * x_2) +
                        x_1 * lambda_21 / (x_2 + lambda_21 * x_1)
                    )

        gamma_1 = np.exp(ln_gamma_1)
        gamma_2 = np.exp(ln_gamma_2)

        return gamma_1, gamma_2
    

    def objective_function_elementwise(self, 
                                       theta: np.ndarray,
                                       regression_params) -> float:

        """
        Objective function for elementwise estimation of Wilson BIP parameters 
        """

        lambda_12 = theta[0]
        lambda_21 = theta[1]

        x_exp_data = regression_params['x1']
        y_exp_data = regression_params['y1']
        pressure_Pa_data = regression_params['pressure_Pa']
        temperature_K = regression_params['temperature_K']
        saturation_pressure_Pa_1 = regression_params['saturation_pressure_Pa_1']
        saturation_pressure_Pa_2 = regression_params['saturation_pressure_Pa_2']
        eos_backed = regression_params['eos_backend']

        error_data = []
        for k in range(len(x_exp_data)):
            x_exp_val = x_exp_data[k]
            y_exp_val = y_exp_data[k]
            pressure_Pa = pressure_Pa_data[k]
            gamma_1_calc, gamma_2_calc = self.get_activity_coefs(lambda_12 = lambda_12,
                                                                 lambda_21 = lambda_21,
                                                                 x_val = x_exp_val)

            if eos_backed is not None:
                fugacity_coef_1, fugacity_coef_2  = eos_backed.get_fugacity_coefs(temperature_K = temperature_K,
                                                                  pressure_Pa = pressure_Pa,
                                                                  molar_composition = np.array([y_exp_val, 1 - y_exp_val])) 
            else: 
                fugacity_coef_1, fugacity_coef_2  = np.array([1.0, 1.0])   # ideal gas assumption if no EoS backend is specified


            # based on the modified Raoult's law: y_i * fi_i * P_total= x_i * gamma_i * P_sat_i 
            y_calc_val_1 = x_exp_val * gamma_1_calc * saturation_pressure_Pa_1 / (pressure_Pa * fugacity_coef_1)
            y_calc_val_2 = (1 - x_exp_val) * gamma_2_calc * saturation_pressure_Pa_2 / (pressure_Pa * fugacity_coef_2)
            
            pressure_total_Pa_calc = ((x_exp_val * gamma_1_calc * saturation_pressure_Pa_1) / (y_calc_val_1 * fugacity_coef_1) + 
                                        ((1 - x_exp_val) * gamma_2_calc * saturation_pressure_Pa_2) / (y_calc_val_2 * fugacity_coef_2))/2

            y_exp_val_1  = y_exp_val
            y_exp_val_2  = 1 - y_exp_val

            error = ((y_exp_val_1 - y_calc_val_1)**2 + 
                     (y_exp_val_2 - y_calc_val_2)**2 + 
                     ((pressure_Pa - pressure_total_Pa_calc)/pressure_Pa)**2)
            error_data.append(error)

        return sum(error_data)
    

    @staticmethod
    def initial_guess_elementwise(initial_guess) -> np.ndarray:
        
        """
        Method to specify initial guess input for the elementwise estimation of Wilson BIPs
        """

        if initial_guess is None:
            return np.array([1.0, 1.0])
        else:
            return initial_guess
        
    
    @staticmethod
    def get_bounds_elementwise() -> tuple:

        """
        Method to get bounds for elementwise estimation of Wilson BIP
        """

        Lambda_12_bounds = (1e-3, None)
        Lambda_21_bounds = (1e-3, None)
        
        bounds = (Lambda_12_bounds, Lambda_21_bounds)

        return bounds


    @staticmethod
    def get_message_elementwise(regression_params: dict,
                                components: list[str],
                                result) -> None:
        
        """
        Method to print a interim results of the elementwise estimation of Wilson activity coefficients
        """

        temperature_K = regression_params['temperature_K']
        pressure_atm  = np.divide(regression_params['pressure_Pa'], 1e5)
        x1_val = regression_params['x1']
        y1_val = regression_params['y1']

        if result.success:

            if len(pressure_atm) > 1:
                press_msg = f"P range = {min(pressure_atm):.2f} - {max(pressure_atm):.2f} atm, "
                comp_msg  = f"x_{components[0]} range = {min(x1_val):.3f} - {max(x1_val):.3f}, " \
                            f"y_{components[0]} range = {min(y1_val):.3f} - {max(y1_val):.3f}"
            elif len(pressure_atm) == 1:
                press_msg = f"P = {pressure_atm[0]:.2f} atm, "
                comp_msg  = f"x_{components[0]} = {x1_val[0]:.3f}, y_{components[0]} = {y1_val[0]:.3f}"
            else:
                raise ValueError(" Either pressure of component data array is empty. ")

            msg = (f"T = {temperature_K:.2f} K, " + 
                   press_msg + 
                   comp_msg + "\n" +
                   " --> Fitted Wilson BIP parameters: "
                   f" lambda_12 = {result.x[0]:.4f}, "
                   f" lambda_21 = {result.x[1]:.4f}. "
                   f" residual = {result.fun:.4e}.")
            
        else:
            msg = (f" BIP parameters regression did not converge: "
                   f" {result.message} ")

        print(msg)

    
    @staticmethod
    def get_polynomial_coeffs_estimation_message(components: list[str],
                                                 estimation_results: list) -> None:
        
        msg = (f"\n DIPPR 4th order polynomial regression of Wilson BIP parameters converged successfully. \n"
               f" Fitted coefficients for Lambda_12: A = {estimation_results[0].x[0]:.4f}, B = {estimation_results[0].x[1]:.4f}, "
               f"C = {estimation_results[0].x[2]:.4f}, D = {estimation_results[0].x[3]:.4f}."
               f" Residual = {estimation_results[0].fun:.4e}. "
               f"\n Fitted coefficients for Lambda_21: A = {estimation_results[1].x[0]:.4f}, B = {estimation_results[1].x[1]:.4f}, "
               f"C = {estimation_results[1].x[2]:.4f}, D = {estimation_results[1].x[3]:.4f}. "
               f" Residual = {estimation_results[1].fun:.4e}. ")
        print(colored(msg, 'green'))


        pass