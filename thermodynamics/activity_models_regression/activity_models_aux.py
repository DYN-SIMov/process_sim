import numpy as np

from typing import Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod

import sys
import os

from termcolor import colored
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import WilsonActivityModel, NRTLActivityModel

@dataclass
class BinaryInteractionParameter():
    name: str
    value: float
    initial_guess: float
    bounds: tuple
    is_temperature_dependant: bool
    is_regressed: bool 


class ActivityModelRegressionInterface(Protocol):

    @staticmethod
    def get_activity_coefs(*args, **kwargs) -> np.ndarray:
        pass

    @staticmethod
    def get_message_elementwise(regression_params: dict,
                                components: list[str],
                                result) -> None:
        pass


class RegressionAuxiliariesMixin(): 


    def get_BIP_names(self) -> list[str]:
        names = []
        for bip in self.BIPs:
            names.append(bip.name)
        return names


    def initial_guess_elementwise(self, initial_guess) -> np.ndarray:
        
        """
        Method to specify initial guess input for the elementwise estimation of Wilson BIPs
        """

        if initial_guess is not None: 
            return initial_guess
        
        initial_guess = [] 
        for bip in self.BIPs:
            if bip.is_regressed and bip.is_temperature_dependant:
                initial_guess.append(bip.initial_guess)
        return np.array(initial_guess)
        
    
    def get_bounds_elementwise(self) -> tuple:

        """
        Method to get bounds for elementwise estimation of Wilson BIP
        """

        bounds_list = []
        for bip in self.BIPs:
            if bip.is_regressed and bip.is_temperature_dependant:
                bounds_list.append(bip.bounds)

        bounds_tuple = tuple(bounds_list)

        return bounds_tuple
    

    def get_temp_dependant_BIPs(self) -> list[BinaryInteractionParameter]:

        """
        Method to get list of temperature dependant regressed parameters for Wilson activity model
        """

        temp_dependant_BIPs = []
        for bip in self.BIPs:
            if bip.is_regressed and bip.is_temperature_dependant:
                temp_dependant_BIPs.append(bip)

        return temp_dependant_BIPs
    

    def get_temp_independant_BIPs(self) -> list[BinaryInteractionParameter]:

        """
        Method to get list of temperature independant regressed parameters for Wilson activity model
        """

        temp_independant_BIPs = []
        for bip in self.BIPs:
            if bip.is_regressed and not bip.is_temperature_dependant:
                temp_independant_BIPs.append(bip)

        return temp_independant_BIPs
    

    def get_bip_index_map(self) -> dict:
        """
        Creates a map of index positions for parameters as they will 
        appear in the 1D regression vector.
        """
        index_map = {}
        idx = 0
        for bip in self.BIPs:
            if bip.is_regressed:
                index_map[bip.name] = idx
                idx += 1
        return index_map
        
    

class WilsonActivityModelRegression(WilsonActivityModel, RegressionAuxiliariesMixin):
    

    def __init__(self,
                 components,
                 pure_component_data_backend):
        super().__init__(components=components,
                         pure_component_data_backend=pure_component_data_backend)
        
        lambda_12 = BinaryInteractionParameter(
            name='Lambda_12',
            value=None,
            initial_guess=1.0,
            bounds=(1e-3, None),
            is_temperature_dependant=True,
            is_regressed=True
        )
        lambda_21 = BinaryInteractionParameter(
            name='Lambda_21',
            value=None,
            initial_guess=1.0,
            bounds=(1e-3, None),
            is_temperature_dependant=True,
            is_regressed=True
        )

        self.BIPs = [lambda_12, lambda_21]
        self._BIP_index_map = self.get_bip_index_map()

        pass


    def get_activity_coefs(self, 
                           BIP_coeffs: np.ndarray,
                           x_val: np.ndarray) -> np.ndarray:

        """
        Method to get activity coeffcients (gamma_1 and gamma_2) based on Wilson equation
        """

        lambda_12 = BIP_coeffs[self._BIP_index_map['Lambda_12']]
        lambda_21 = BIP_coeffs[self._BIP_index_map['Lambda_21']]

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
    



class NRTLActivityModelRegression(NRTLActivityModel, RegressionAuxiliariesMixin): 

    def __init__(self,
                 components,
                 pure_component_data_backend,
                 alpha_is_fixed:bool = True,
                 alpha:float = 0.3):
        super().__init__(components=components,
                         pure_component_data_backend=pure_component_data_backend)
        
        self.alpha_is_fixed = alpha_is_fixed
        self.alpha = alpha
        
        tau_12 = BinaryInteractionParameter(
            name='tau_12',
            value=None,
            initial_guess=1.0,
            bounds=(1e-3, None),
            is_temperature_dependant=True,
            is_regressed=True
        )

        tau_21 = BinaryInteractionParameter(
            name='tau_21',
            value=None,
            initial_guess=1.0,
            bounds=(1e-3, None),
            is_temperature_dependant=True,
            is_regressed=True
        )

        alpha_12 = BinaryInteractionParameter(
            name='alpha_12',
            value=alpha if alpha_is_fixed else None,
            initial_guess=alpha,
            bounds=None if alpha_is_fixed else (1e-3, 0.5),
            is_temperature_dependant=False,
            is_regressed=False if alpha_is_fixed else True
        )

        alpha_21 = BinaryInteractionParameter(
            name='alpha_21',
            value=alpha if alpha_is_fixed else None,
            initial_guess=alpha,
            bounds=None if alpha_is_fixed else (1e-3, 0.5),
            is_temperature_dependant=False,
            is_regressed=False if alpha_is_fixed else True
        )

        self.BIPs = [tau_12, tau_21, alpha_12, alpha_21]
        self._BIP_index_map = self.get_bip_index_map()
        



    def get_activity_coefs(self, 
                           BIP_coeffs: np.ndarray,
                           x_val: np.ndarray) -> np.ndarray:

        """
        Method to get activity coeffcients (gamma_1 and gamma_2) based on NRTL equation
        """

        tau_12 = BIP_coeffs[self._BIP_index_map['tau_12']]
        tau_21 = BIP_coeffs[self._BIP_index_map['tau_21']]

        if self.alpha_is_fixed:
            alpha_12 = self.alpha
            alpha_21 = self.alpha
        else:
            alpha_12 = BIP_coeffs[self._BIP_index_map['alpha_12']]
            alpha_21 = BIP_coeffs[self._BIP_index_map['alpha_21']]

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
    
    
    def get_message_elementwise(self,
                                regression_params: dict,
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

            if self.alpha_is_fixed:
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
                    f" residual = {result.fun:.4e}.")
            
        else:
            msg = (f" BIP parameters regression did not converge: "
                   f" {result.message} ")

        print(msg)

    
    def get_polynomial_coeffs_estimation_message(self, estimation_results: list) -> None:
        
        if self.alpha_is_fixed:
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


    def get_message_estimation_from_VLE(self, coeffs: list,
                                        total_residual: float) -> None:
        
        if self.alpha_is_fixed:
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

