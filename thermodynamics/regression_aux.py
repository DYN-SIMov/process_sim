import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")                       # switch off interactive mode (suppresssing the warning messages)
import matplotlib.pyplot as plt 

from typing import Protocol

from itertools import takewhile             # for parsing
from scipy.optimize import minimize         # for regression 
from termcolor import colored               # for colored text output

from CoolProp.CoolProp import PropsSI
from chemicals import CAS_from_any

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from properties import SoaveRedlichKwongEoSBackend, PureComponentDataBackend


class ActivityModelInterface(Protocol):
    def activity_coefs(self, x, T, params):
        ...


class WilsonActivityModel():
    
    @staticmethod
    def get_activity_coefs(lambda_12: float,
                           lambda_21: float,
                           x_val: np.ndarray) -> np.ndarray:

        " Returns arrays of activity coefficients based on Wilson equations. "

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
    

    def objective_function(self, 
                           theta: np.ndarray,
                           regression_params) -> float:

        " Objective function to minimize for Wilson BIP parameters estimation. "

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

            y_exp_val_1  = y_exp_val
            y_exp_val_2  = 1 - y_exp_val

            error = (y_exp_val_1 - y_calc_val_1)**2 + (y_exp_val_2 - y_calc_val_2)**2
            error_data.append(error)

        return sum(error_data)
    

    @staticmethod
    def initial_guess(initial_guess) -> np.ndarray:
        
        if initial_guess is None:
            return np.array([1.0, 1.0])
        else:
            return initial_guess
        

    @staticmethod
    def get_message(regression_params: dict,
                    components: list[str],
                    result) -> None:
        
        temperature_K = regression_params['temperature_K']
        pressure_atm  = regression_params['pressure_Pa'] / 1e5
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




class NRTLActivityModel():
    def activity_coefs(self, x, T, params):
        # Implement NRTL model equations here
        pass



class BinaryInteractionParametersRegression(): 

    def __init__(self,
                 activity_model: str = None,
                 equation_of_state: str = None) -> None: 

        self.activity_model = activity_model
        self.equation_of_state = equation_of_state


    def _parse_components_from_comment(self, filepath: str) -> list[str]:

        "Function to parse component names from comment lines in the data file. "

        components = []
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("# Component"):
                    comment_line = line.strip()
                    components.append(comment_line.split()[-1])
        
        # checking that exactly two components are provided
        if len(components) != 2:
            raise ValueError(" Binary interaction parameters regression requires exactly two components. ")
        
        return components
        
    
    def _extract_data(self, dataframe: pd.DataFrame, components: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        " Function to extract process condition data and composition data from the dataframe. "

        # exctracting process condition data
        pressure_Pa_data   = dataframe["P (atm)"].to_numpy() * 1e5    # converting from bar to Pa
        temperature_K_data = dataframe["T (degC)"].to_numpy() + 273.15
        
        # exctracting componens fraction data
        for comp in components:
            col_x = f"x_{comp}"
            col_y = f"y_{comp}"
            if col_x in dataframe.columns and col_y in dataframe.columns:
                x_data = dataframe[col_x].to_numpy()
                y_data = dataframe[col_y].to_numpy()
                break
        else:
            raise KeyError("No matching x_* / y_* columns found for given components")
        
        return pressure_Pa_data, temperature_K_data, x_data, y_data
    

    def _detect_data(self, 
                     data:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        
        """
        Detects if the VLE data are isobaric (T-x-y), isothermal (P-x-y), or mixed.
        If mixed, splits into isobaric and isothermal sets.
        Stores result in self.data_sets.
        """

        self.data_set = []
        pressure_Pa, temperature_K, x_data, y_data = data
        tol = 1e-2  # tolerance for "constant" (Pa or K)

        def is_constant(arr):
            # tests if all values in arr are the same within tolerance
            # catches arrays with too few data points
            if len(arr) <= 1:
                raise ValueError("Only one data point provided, cannot proceed with regression.")
            return np.all(np.abs(arr - arr[0]) < tol)


        if is_constant(pressure_Pa):
            # Isobaric data 
            for k in range(len(temperature_K)): 
                self.data_set.append({
                    'type': 'isobaric_single_point',
                    'pressure_Pa': np.array([pressure_Pa[k]]),
                    'temperature_K': temperature_K[k],
                    'x_data': np.array([x_data[k]]),
                    'y_data': np.array([y_data[k]]),
                    'indices': np.array([k])
                })

            msg =(f" Data import: "
                  f" Detected isobaric VLE data set for {self.components[0]} and {self.components[1]}" 
                  f" at P = {pressure_Pa[0]/1e5:.2f} bar. ")
            print(colored(msg, 'green'))

        elif is_constant(temperature_K):
            # Isothermal data set
            self.data_set.append({
                    'type': 'isothermal',
                    'pressure_Pa': np.array(pressure_Pa),
                    'temperature_K': temperature_K[0],
                    'x_data': np.array([x_data]),
                    'y_data': np.array([y_data]),
                    'indices': np.array([k for k in range(len(temperature_K))])
                })

            msg =(f" Data import warning: "
                  f" Detected only one isothermal VLE data set for {self.components[0]} and {self.components[1]} " 
                  f" at T = {temperature_K[0]:.2f} K. \n"
                  f" Regression is likely to be inaccurate due to lack of temperature variability." 
                  f" Consider adding more data points at different temperatures. ")
            print(colored(msg, 'yellow'))
            
        else:
            # Mixed data set - split into isobaric and isothermal
            unique_temperatures = np.unique(temperature_K)
            for temperature_val in unique_temperatures:
                    indices = np.where(np.abs(temperature_K - temperature_val) < tol)[0]
                    self.data_set.append({
                        'type': 'isothermal',
                        'pressure_Pa': pressure_Pa[indices],
                        'temperature_K': temperature_val,
                        'x_data': x_data[indices],
                        'y_data': y_data[indices],
                        'indices': indices
                    })

            msg = (f" Data import: "
                   f" Detected mixed VLE data set for {self.components[0]} and {self.components[1]}. " 
                   f" Split into {len(unique_temperatures)} isothermal subsets. ")
            print(colored(msg, 'green'))

        
    def _configure_activity_model_backend(self,
                                          components: list) -> None:

        if self.activity_model is not None:
            if self.activity_model.upper() == 'WILSON':
                self.activity_model_backend = WilsonActivityModel()
            elif self.activity_model.upper() == 'NRTL':
                self.activity_model_backend = NRTLActivityModel()
            else:
                raise ValueError(" Unsupported activity model specified for BIP parameters regression. ")
            
        else: 
            self.activity_model_backend = WilsonActivityModel()
            warning_msg = " No activity model specified for BIP parameters regression. " \
                          " Wilson activity model will be used by default. " 
            print(colored(warning_msg, 'yellow'))

        pass


    def _configure_eos_backend(self,
                               components: list) -> None:

        if self.equation_of_state is not None:
            if self.equation_of_state.upper() in ['SRK', 'SOAVE-REDLICH-KWONG', 'SOAVE REDLICH KWONG']:
                pure_component_data_backend = PureComponentDataBackend(components = components)
                self.eos_backend = SoaveRedlichKwongEoSBackend(components=components,
                                                               pure_component_data_backend=pure_component_data_backend)

                pass

            elif self.equation_of_state.upper() in ['PR', 'PENG-ROBERTSON', 'PENG ROBERTSON']:
                raise NotImplementedError(" Peng-Robertson EoS backend for BIP parameters regression is not implemented yet. ")
            else:
                raise ValueError(" Unsupported equation of state specified for BIP parameters regression. ")

        pass 


    def data_import(self, filepath: str) -> None:

        if filepath is None: 
            raise ValueError(" No filepath specified for data import. ")

        df = pd.read_csv(filepath_or_buffer = filepath, comment='#')

        components = self._parse_components_from_comment(filepath = filepath)
        self.components = components

        data = self._extract_data(dataframe = df, components = self.components)

        self._detect_data(data = data)
        self._configure_activity_model_backend(components=components)
        self._configure_eos_backend(components=components)

        pass

    
    def regress_BIP_parameters_elementwise(self) -> None:
        " Function to regress Wilson BIP parameters for each temperature point (i.e., data_set) individually. "

        opt_data_results = []

        for data_set in self.data_set:

            try: 
                saturation_pressure_Pa_1 = PropsSI('P','T',data_set['temperature_K'],'Q',1,CAS_from_any(self.components[0]))
                saturation_pressure_Pa_2 = PropsSI('P','T',data_set['temperature_K'],'Q',1,CAS_from_any(self.components[1]))
            except:
                msg = (f" Saturation pressure calculation failed for T = {data_set['temperature_K']:.2f} K. "
                        f" Skipping this data point." 
                        f" Check if critical temperature is exceeded or if components are not supported by CoolProp. ")
                print(colored(msg, 'yellow'))
                break

            regression_params = {'y1': data_set['y_data'],
                                    'x1': data_set['x_data'],
                                    'saturation_pressure_Pa_1': saturation_pressure_Pa_1,
                                    'saturation_pressure_Pa_2': saturation_pressure_Pa_2,
                                    'pressure_Pa': data_set['pressure_Pa'],
                                    'temperature_K': data_set['temperature_K'],
                                    'eos_backend': self.eos_backend if self.equation_of_state is not None else None}

            result = minimize(fun = self.activity_model_backend.objective_function,
                              x0 = self.activity_model_backend.initial_guess(initial_guess=opt_data_results[-1] 
                                                                            if len(opt_data_results) > 0 else None),
                              args = (regression_params),
                              method = 'SLSQP',
                              bounds=((1e-3, None), (1e-3, None))) # NOTE define dedicated bounds based on activity model

            msg = self.activity_model_backend.get_message(result=result,
                                                            regression_params=regression_params,
                                                            components=self.components)

            opt_data_results.append(result.x)

        self.elementwise_opt_results = np.array(opt_data_results)

        pass


    @ staticmethod
    def _DIPPR_4p_polynomial_Wilson(coeffs,
                                    temperature_K) -> float:

        " Polynomial for BIP: Lambda_ij = exp(A + B/T + C*ln(T) + D*T). "
        " Based on simplified version of the polynomial form described in docs "
        " of thermo python library (thermo.Wilson). "

        A, B, C, D = coeffs
        T = temperature_K

        Lambda_ij_calc = np.exp(A + B/T + C*np.log(T) + D*T)

        return Lambda_ij_calc


    def _DIPPR_4p_polynomial_Wilson_objective_function(self, 
                                                       coeffs: np.ndarray,
                                                       temperature_K_data: np.ndarray,
                                                       bip_data: np.ndarray) -> float:
        " Objective function for DIPPR polynomial regression of Wilson BIP parameters. "
        
        error_data = []
        for k in range(len(temperature_K_data)):
            ln_Lambda_ij_exp = np.log(bip_data[k])

            Lambda_ij_calc    = self._DIPPR_4p_polynomial_Wilson(coeffs = coeffs,
                                                                 temperature_K = temperature_K_data[k])
            ln_Lambda_ij_calc = np.log(Lambda_ij_calc)

            error = (ln_Lambda_ij_exp - ln_Lambda_ij_calc)**2
            error_data.append(error)

        return sum(error_data)


    def estimate_DIPPR_polynomial_from_elementwise_optimisation(self) -> None:
        
        if self.elementwise_opt_results is None:
            raise ValueError(" Elementwise optimisation results not found. "
                             " Run regress_BIP_parameters_elementwise() first. ")
        
        esitmation_completed = False

        if self.activity_model.upper() == 'WILSON':
            
            temperature_K_data = []
            lambda_12_data = []
            lambda_21_data = []

            for data_set, opt_result in zip(self.data_set, self.elementwise_opt_results):
                temperature_K_data.append(data_set['temperature_K'])
                lambda_12_data.append(opt_result[0])
                lambda_21_data.append(opt_result[1])

            results_12 = minimize(fun = self._DIPPR_4p_polynomial_Wilson_objective_function,
                                  x0 = np.array([0.0, 0.0, 0.0, 0.0]),
                                  args = (np.array(temperature_K_data), np.array(lambda_12_data)),
                                  method = 'SLSQP')
            
            results_21 = minimize(fun = self._DIPPR_4p_polynomial_Wilson_objective_function,
                                  x0 = np.array([0.0, 0.0, 0.0, 0.0]),
                                  args = (np.array(temperature_K_data), np.array(lambda_21_data)),
                                  method = 'SLSQP')
            
            if results_12.success and results_21.success:
                self.DIPPR_polynomial_coeffs = {
                    'Lambda_12': results_12.x,
                    'Lambda_21': results_21.x
                }
                esitmation_completed = True

                msg = (f"\n DIPPR 4th order polynomial regression of Wilson BIP parameters converged successfully. \n"
                       f" Fitted coefficients for Lambda_12: A = {results_12.x[0]:.4f}, B = {results_12.x[1]:.4f}, "
                       f"C = {results_12.x[2]:.4f}, D = {results_12.x[3]:.4f}."
                       f" Residual = {results_12.fun:.4e}. "
                       f"\n Fitted coefficients for Lambda_21: A = {results_21.x[0]:.4f}, B = {results_21.x[1]:.4f}, "
                       f"C = {results_21.x[2]:.4f}, D = {results_21.x[3]:.4f}. "
                       f" Residual = {results_21.fun:.4e}. ")
                print(colored(msg, 'green'))
            else: 
                raise RuntimeError(" DIPPR polynomial regression of Wilson BIP parameters did not converge. ")

        

        if esitmation_completed is False:
            raise NotImplementedError(" DIPPR polynomial regression not implemented for the specified activity model. ")

        
        pass



    def record_regression_results(self) -> None:
        

        pass



    def plot_regression_results(self) -> None:
        

        pass
