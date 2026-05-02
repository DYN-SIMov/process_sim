import numpy as np

from typing import Protocol

import sys
import os

from termcolor import colored
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import WilsonActivityModel, NRTLActivityModel

from optimization import PolynomialInterface, PolynomialExponentialDIPPR, PolynomialNRTL

class PolynomialException(Exception):
    pass


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
    
    number_of_BIP_parameters:int = 2

    def __init__(self,
                 components,
                 pure_component_data_backend):
        super().__init__(components=components,
                         pure_component_data_backend=pure_component_data_backend)
        pass


    @staticmethod
    def get_activity_coefs(theta: np.ndarray,
                           x_val: np.ndarray) -> np.ndarray:

        """
        Method to get activity coeffcients (gamma_1 and gamma_2) based on Wilson equation
        """

        lambda_12 = theta[0]
        lambda_21 = theta[1]

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
    def get_polynomial_coeffs_estimation_message(estimation_results: list) -> None:
        
        msg = (f"\n Polynomial regression of Wilson BIP parameters converged successfully. \n"
               f" Fitted coefficients for Lambda_12: A = {estimation_results[0].x[0]:.4f}, "
               f"B = {estimation_results[0].x[1]:.4f}, C = {estimation_results[0].x[2]:.4f}, "
               f"D = {estimation_results[0].x[3]:.4f}. Residual = {estimation_results[0].fun:.4e}. "
               f"\n Fitted coefficients for Lambda_21: A = {estimation_results[1].x[0]:.4f}, " 
               f"B = {estimation_results[1].x[1]:.4f}, C = {estimation_results[1].x[2]:.4f}, " 
               f"D = {estimation_results[1].x[3]:.4f}. Residual = {estimation_results[1].fun:.4e}. ")
        print(colored(msg, 'green'))


        pass


    @staticmethod
    def get_message_estimation_from_VLE(coeffs: list,
                                        total_residual: float) -> None:
        
        msg = (f"\n DIPPR 4th order polynomial regression of Wilson BIP parameters converged successfully. \n"
               f" Fitted coefficients for Lambda_12: A = {coeffs[0]:.4f}, "
               f"B = {coeffs[1]:.4f}, C = {coeffs[2]:.4f}, D = {coeffs[3]:.4f}. "
               f"\n Fitted coefficients for Lambda_21: A = {coeffs[4]:.4f}, " 
               f"B = {coeffs[5]:.4f}, C = {coeffs[6]:.4f}, D = {coeffs[7]:.4f}. "
               f"\n Total residual = {total_residual:.4e}. ")
        print(colored(msg, 'green'))


        pass


    @staticmethod
    def export_check_polynomial_type(polynomial: PolynomialInterface) -> None:
        
        if not isinstance(polynomial, PolynomialExponentialDIPPR):
            raise PolynomialException(
                f" The specified polynomial {type(polynomial)} is not compatible "
                f"with Wilson activity model regression. \n"
                f" Please use {PolynomialExponentialDIPPR} polynomial for regression of "
                f"Wilson BIP parameters. "
            )

    @staticmethod
    def get_data_for_export(): 


        pass



class NRTLActivityModelRegression(NRTLActivityModel): 

    alpha_is_fixed:bool = True
    alpha:float = 0.3 

    number_of_BIP_parameters:int = 2 if alpha_is_fixed else 4

    def __init__(self,
                 components,
                 pure_component_data_backend):
        super().__init__(components=components,
                         pure_component_data_backend=pure_component_data_backend)
        pass


    @staticmethod
    def get_activity_coefs(theta: np.ndarray,
                           x_val: np.ndarray) -> np.ndarray:

        """
        Method to get activity coeffcients (gamma_1 and gamma_2) based on NRTL equation
        """

        tau_12 = theta[0]
        tau_21 = theta[1]

        if NRTLActivityModelRegression.alpha_is_fixed:
            alpha_12 = NRTLActivityModelRegression.alpha
            alpha_21 = NRTLActivityModelRegression.alpha
        else:
            alpha_12 = theta[2]
            alpha_21 = theta[3]

        x_1 = x_val
        x_2 = 1 - x_val

        G_12 = np.exp(-alpha_12 * tau_12)
        G_21 = np.exp(-alpha_21 * tau_21)

        ln_gamma_1 = x_2**2 * (tau_21 * (G_21 / (x_1 + x_2 * G_21))**2 + 
                               tau_12 * G_12 / (x_2 + x_1 * G_12)**2)
        ln_gamma_2 = x_1**2 * (tau_12 * (G_12 / (x_2 + x_1 * G_12))**2 + 
                               tau_21 * G_21 / (x_1 + x_2 * G_21)**2)
        
        gamma_1 = np.exp(ln_gamma_1)
        gamma_2 = np.exp(ln_gamma_2)

        return gamma_1, gamma_2


    @staticmethod
    def initial_guess_elementwise(initial_guess) -> np.ndarray:
        
        """
        Method to specify initial guess input for the elementwise estimation of NRTL BIPs
        """

        if initial_guess is None:
            if NRTLActivityModelRegression.alpha_is_fixed:
                return np.array([1.0, 1.0])
            else:
                return np.array([1.0, 1.0, 0.3, 0.3])
        else:
            return initial_guess
        

    @staticmethod
    def get_bounds_elementwise() -> tuple:

        """
        Method to get bounds for elementwise estimation of NRTL BIP
        """

        tau_12_bounds = (1e-3, None)
        tau_21_bounds = (1e-3, None)
        
        if NRTLActivityModelRegression.alpha_is_fixed:
            bounds = (tau_12_bounds, tau_21_bounds)
        else:
            alpha_12_bounds = (1e-3, None)
            alpha_21_bounds = (1e-3, None)
            bounds = (tau_12_bounds, tau_21_bounds, alpha_12_bounds, alpha_21_bounds)

        return bounds
    
    
    @staticmethod
    def get_message_elementwise(regression_params: dict,
                                components: list[str],
                                result) -> None:
        
        """
        Method to print a interim results of the elementwise estimation of NRTL activity coefficients
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

            if NRTLActivityModelRegression.alpha_is_fixed:
                msg = (f"T = {temperature_K:.2f} K, " + 
                    press_msg + 
                    comp_msg + "\n" +
                    " --> Fitted NRTL BIP parameters: "
                    f" tau_12 = {result.x[0]:.4f}, "
                    f" tau_21 = {result.x[1]:.4f}. "
                    f" residual = {result.fun:.4e}.")
            else:
                msg = (f"T = {temperature_K:.2f} K, " + 
                    press_msg + 
                    comp_msg + "\n" +
                    " --> Fitted NRTL BIP parameters: "
                    f" tau_12 = {result.x[0]:.4f}, "
                    f" tau_21 = {result.x[1]:.4f}, "
                    f" alpha_12 = {result.x[2]:.4f}, "
                    f" alpha_21 = {result.x[3]:.4f}. "
                    f" residual = {result.fun:.4e}.")
            
        else:
            msg = (f" BIP parameters regression did not converge: "
                   f" {result.message} ")

        print(msg)

    
    @staticmethod
    def get_polynomial_coeffs_estimation_message(estimation_results: list) -> None:
        
        if NRTLActivityModelRegression.alpha_is_fixed:
            msg = (
                f"\n Polynomial regression of NRTL BIP parameters converged successfully. \n"
                f" Fitted coefficients for tau_12: A = {estimation_results[0].x[0]:.4f}, "
                f"B = {estimation_results[0].x[1]:.4f}, C = {estimation_results[0].x[2]:.4f}, "
                f"D = {estimation_results[0].x[3]:.4f}. Residual = {estimation_results[0].fun:.4e}. "
                f"\n Fitted coefficients for tau_21: A = {estimation_results[1].x[0]:.4f}, " 
                f"B = {estimation_results[1].x[1]:.4f}, C = {estimation_results[1].x[2]:.4f}, " 
                f"D = {estimation_results[1].x[3]:.4f}. Residual = {estimation_results[1].fun:.4e}. "
            )
        else: 
            msg = (
                " MESSAGE FOR NRTL POLYNOMIAL REGRESSION WITH VARIABLE ALPHA IS NOT IMPLEMENTED YET. "
            )

        print(colored(msg, 'green'))


        pass


    @staticmethod
    def get_message_estimation_from_VLE(coeffs: list,
                                        total_residual: float) -> None:
        
        if NRTLActivityModelRegression.alpha_is_fixed:
            msg = (
                f"\n Polynomial regression of NRTL BIP parameters converged successfully. \n"
                f" Fitted coefficients for tau_12: A = {coeffs[0]:.4f}, "
                f"B = {coeffs[1]:.4f}, C = {coeffs[2]:.4f}, D = {coeffs[3]:.4f}. "
                f"\n Fitted coefficients for tau_21: A = {coeffs[4]:.4f}, " 
                f"B = {coeffs[5]:.4f}, C = {coeffs[6]:.4f}, D = {coeffs[7]:.4f}. "
                f"\n Total residual = {total_residual:.4e}. "
            )
        else: 
            msg = (
                " MESSAGE FOR NRTL POLYNOMIAL REGRESSION WITH VARIABLE ALPHA IS NOT IMPLEMENTED YET. "
            )
        
        print(colored(msg, 'green'))

        pass


    @staticmethod
    def export_check_polynomial_type(polynomial: PolynomialInterface) -> None:
        
        if not isinstance(polynomial, PolynomialNRTL):
            raise PolynomialException(
                f" The specified polynomial {type(polynomial)} is not compatible "
                f"with NRTL activity model regression. \n"
                f" Please use {PolynomialNRTL} polynomial for regression of "
                f"NRTL BIP parameters. "
            )
