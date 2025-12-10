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

        x_exp_val = regression_params['x1_val']
        y_exp_val = regression_params['y1_val']
        saturation_pressure_Pa_1 = regression_params['saturation_pressure_Pa_1']
        saturation_pressure_Pa_2 = regression_params['saturation_pressure_Pa_2']
        pressure_Pa = regression_params['pressure_Pa']
        fugacity_coef_1 = regression_params['fugacity_coefs'][0]
        fugacity_coef_2 = regression_params['fugacity_coefs'][1]

        gamma_1_calc, gamma_2_calc = self.get_activity_coefs(lambda_12 = lambda_12,
                                                             lambda_21 = lambda_21,
                                                             x_val = x_exp_val)

        # based on the modified Raoult's law: y_i * fi_i * P_total= x_i * gamma_i * P_sat_i 
        y_calc_val_1 = x_exp_val * gamma_1_calc * saturation_pressure_Pa_1 / (pressure_Pa * fugacity_coef_1)
        y_calc_val_2 = (1 - x_exp_val) * gamma_2_calc * saturation_pressure_Pa_2 / (pressure_Pa * fugacity_coef_2)

        y_exp_val_1  = y_exp_val
        y_exp_val_2  = 1 - y_exp_val

        error = (y_exp_val_1 - y_calc_val_1)**2 + (y_exp_val_2 - y_calc_val_2)**2

        return error
    

    @staticmethod
    def initial_guess(initial_guess) -> np.ndarray:
        
        if initial_guess is None:
            return np.array([1.0, 1.0])
        else:
            return initial_guess
        

    @staticmethod
    def get_message(regression_params: dict,
                    components: list[str],
                    result) -> str:
        
        temperature_K = regression_params['temperature_K']
        pressure_atm  = regression_params['pressure_Pa'] / 1e5
        x1_val = regression_params['x1_val']
        y1_val = regression_params['y1_val']

        if result.success:
            msg = (f"T = {temperature_K:.2f} K, P = {pressure_atm:.2f} atm, "
                   f"x_{components[0]} = {x1_val:.3f}, y_{components[0]} = {y1_val:.3f} --> "
                   f" Fitted Wilson BIP parameters: "
                   f" lambda_12 = {result.x[0]:.6f}, "
                   f" lambda_21 = {result.x[1]:.6f}. "
                   f" residual = {result.fun:.6e}.")
        else:
            msg = (f" BIP parameters regression did not converge: "
                   f" {result.message} ")

        return msg




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
            return np.all(np.abs(arr - arr[0]) < tol)
        
        if is_constant(pressure_Pa):
            # Isobaric data set
            self.data_set.append({
                'type': 'isobaric',
                'pressure_Pa': pressure_Pa,
                'temperature_K': temperature_K,
                'x_data': x_data,
                'y_data': y_data,
                'indices': np.arange(len(x_data))
            })

            msg =(f" Data import: "
                  f" Detected isobaric VLE data set for {self.components[0]} and {self.components[1]} " 
                  f"at P = {pressure_Pa[0]/1e5:.2f} bar. ")

        elif is_constant(temperature_K):
            # Isothermal data set
            self.data_set.append({
                'type': 'isothermal',
                'pressure_Pa': pressure_Pa,
                'temperature_K': temperature_K,
                'x_data': x_data,
                'y_data': y_data,
                'indices': np.arange(len(x_data))
            })

            msg =(f" Data import: "
                  f" Detected isothermal VLE data set for {self.components[0]} and {self.components[1]} " 
                  f" at T = {temperature_K[0]:.2f} K. ")

        else:

            # Mixed data set - split into isobaric and isothermal
            unique_pressures = np.unique(pressure_Pa)
            unique_temperatures = np.unique(temperature_K)

            if len(unique_pressures) < len(unique_temperatures):
                # More unique temperatures - treat as isobaric sets
                for pressure_val in unique_pressures:
                    indices = np.where(np.abs(pressure_Pa - pressure_val) < tol)[0]
                    self.data_set.append({
                        'type': 'isobaric',
                        'pressure_Pa': pressure_Pa[indices],
                        'temperautre_K': temperature_K[indices],
                        'x_data': x_data[indices],
                        'y_data': y_data[indices],
                        'indices': indices
                    })

                msg =(f" Data import: "
                      f" Detected mixed VLE data set for {self.components[0]} and {self.components[1]}. "
                      f" Split into {len(unique_pressures)} isobaric subsets. ")

            else:
                # More unique pressures - treat as isothermal sets
                for temperature_val in unique_temperatures:
                    indices = np.where(np.abs(temperature_K - temperature_val) < tol)[0]
                    self.data_set.append({
                        'type': 'isothermal',
                        'pressure_Pa': pressure_Pa[indices],
                        'temperature_K': temperature_K[indices],
                        'x_data': x_data[indices],
                        'y_data': y_data[indices],
                        'indices': indices
                    })

                msg = (f" Data import: "
                       f" Detected mixed VLE data set for {self.components[0]} and {self.components[1]}. " 
                       f" Split into {len(unique_temperatures)} isothermal subsets. ")

        if msg:        
            print(colored(msg, 'green'))
        else:
            raise print(colored(" Data import: Unable to detect VLE data type (isobaric/isothermal/mixed). ",'red'))
        

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
        
        opt_data_results = []
        opt_data_temperature_K = []
        opt_data_pressure_Pa = []

        for data_set in self.data_set:
            for k in range(len(data_set['x_data'])):

                try: 
                    saturation_pressure_Pa_1 = PropsSI('P','T',data_set['temperature_K'][k],'Q',1,CAS_from_any(self.components[0]))
                    saturation_pressure_Pa_2 = PropsSI('P','T',data_set['temperature_K'][k],'Q',1,CAS_from_any(self.components[1]))
                except:
                    msg = (f" Saturation pressure calculation failed for T = {data_set['temperature_K'][k]:.2f} K. "
                           f" Skipping this data point." 
                           f" Check if critical temperature is exceeded or if components are not supported by CoolProp. ")
                    print(colored(msg, 'yellow'))
                    break


                if self.equation_of_state is not None:
                    fugacity_coefs    = self.eos_backend.get_fugacity_coefs(temperature_K = data_set['temperature_K'][k],
                                                                            pressure_Pa = data_set['pressure_Pa'][k],
                                                                            molar_composition = np.array([data_set['y_data'][k], 1 - data_set['y_data'][k]])) 
                else: 
                    fugacity_coefs    = np.array([1.0, 1.0])   # ideal gas assumption if no EoS backend is specified


                regression_params = {'y1_val': data_set['y_data'][k],
                                     'x1_val': data_set['x_data'][k],
                                     'saturation_pressure_Pa_1': saturation_pressure_Pa_1,
                                     'saturation_pressure_Pa_2': saturation_pressure_Pa_2,
                                     'pressure_Pa': data_set['pressure_Pa'][k],
                                     'temperature_K': data_set['temperature_K'][k],
                                     'fugacity_coefs': fugacity_coefs}

                result = minimize(fun = self.activity_model_backend.objective_function,
                                  x0 = self.activity_model_backend.initial_guess(initial_guess=opt_data_results[-1] 
                                                                                 if len(opt_data_results) > 0 else None),
                                  args = (regression_params),
                                  method = 'SLSQP',
                                  bounds=((1e-3, None), (1e-3, None)))

                msg = self.activity_model_backend.get_message(result=result,
                                                              regression_params=regression_params,
                                                              components=self.components)
                print(msg)

                opt_data_results.append(result.x)
                opt_data_temperature_K.append(data_set['temperature_K'][k])
                opt_data_pressure_Pa.append(data_set['pressure_Pa'][k])
                
            print()  # blank line between data sets

        self.opt_results = np.array(opt_data_results)
        self.opt_temperature_K = np.array(opt_data_temperature_K)
        self.opt_pressure_Pa = np.array(opt_data_pressure_Pa)

        pass




    def estimate_DIPPR_polynomial(self) -> None:
        

        
        pass



    def record_regression_results(self) -> None:
        

        pass



    def plot_regression_results(self) -> None:
        

        pass
