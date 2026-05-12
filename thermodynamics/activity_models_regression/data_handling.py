from dataclasses import dataclass

import pandas as pd
import numpy as np
import re

from termcolor import colored        # for colored text output

from CoolProp.CoolProp import PropsSI
from chemicals import CAS_from_any

from thermo import Chemical

class VLEDataError(Exception):
    pass


@dataclass(frozen=True)
class DataPoint: 

    """
    Represents a single Vapor-Liquid Equilibrium (VLE) experimental data point.
    """
    pressure_Pa: float
    temperature_K: float
    x1_mol_frac: float
    y1_mol_frac: float
    x2_mol_frac: float
    y2_mol_frac: float
    comp1: str
    comp2: str



class TxyPoint(): 
    
    def __init__(self,
                 data_points: list[DataPoint]):
        self.data = data_points
        self.temperature_K: float = data_points[0].temperature_K    
        self.comp_1_saturation_pressure_Pa: float = None
        self.comp_2_saturation_pressure_Pa: float = None

    pass



class RawExperimentalData(list[DataPoint]): 

    def __init__(self,
                 data_points: list[DataPoint] = [],
                 remove_extreme_concentrations: bool = True,
                 concentration_tol: float = 1e-3):
        
        if remove_extreme_concentrations:
            filtered_data_points = []
            for point in data_points:
                if (point.x1_mol_frac >= concentration_tol and 
                    point.x1_mol_frac <= 1-concentration_tol and
                    point.y1_mol_frac >= concentration_tol and 
                    point.y1_mol_frac <= 1-concentration_tol):
                    filtered_data_points.append(point)
                else:
                    msg = (f"Data import [warning]: "
                           f" Removed data point at P = {point.pressure_Pa/1e5:.2f} bar, "
                           f"T = {point.temperature_K:.2f} K due to extreme concentrations "
                           f" x_{point.comp1} = {point.x1_mol_frac:.6f}, "
                           f" y_{point.comp1} = {point.y1_mol_frac:.6f}. ")
                    print(colored(msg, 'yellow'))
            self.data_points = filtered_data_points
        else:
            self.data_points = data_points


    def find_constant_temperature_points(self, temperature_K_tol: float = 1e-3) -> list[TxyPoint]: 
        T_x_y_points = []
        constant_temperature_points = []
        if not self.data_points:
            raise VLEDataError(" No data points available to search for constant temperature points. ")
 
        reference_temperature_data = [self.data_points[0].temperature_K]
        for point in self.data_points:  
            detected_temp_diff = np.abs(np.array(reference_temperature_data) - point.temperature_K)
            if np.all(detected_temp_diff > temperature_K_tol):
                T_x_y_points.append(
                    TxyPoint(data_points = constant_temperature_points) if constant_temperature_points 
                        else TxyPoint(data_points = [point])
                        )
                constant_temperature_points = [point]
                reference_temperature_data.append(point.temperature_K)
            else:
                constant_temperature_points.append(point)
        
        # Add the last batch of constant temperature points
        if len(constant_temperature_points) > 0:
            T_x_y_points.append(TxyPoint(data_points = constant_temperature_points))
                
        if len(T_x_y_points) < 2:
            msg =(f" Data import [warning]: "
                  f" Detected only one isothermal VLE data set for" 
                  f" {constant_temperature_points[1].comp1} and {constant_temperature_points[1].comp2} " 
                  f" at T = {constant_temperature_points[1].temperature_K:.2f} K. \n"
                  f" Regression is likely to be inaccurate due to lack of temperature variability." 
                  f" Consider adding more data points at different temperatures. ")
            print(colored(msg, 'yellow'))

        return T_x_y_points



    pass



class VLEData():
    
    def __init__(self,
                 filepath: str, 
                 remove_extreme_concentrations: bool = True):
        self.filepath = filepath
        self.remove_extreme_concentrations = remove_extreme_concentrations 
        self.components: list[str] = self._parse_components_from_comment(filepath = filepath)
        self.source: str = self._parse_source_from_comment(filepath = filepath)
        self.raw_dataframe: pd.DataFrame = pd.read_csv(filepath_or_buffer = filepath, comment='#')
        self.raw_data: RawExperimentalData = self._extract_data_from_data_frame(dataframe = self.raw_dataframe)
        self.T_x_y_points: list[TxyPoint] = self.raw_data.find_constant_temperature_points()

        self._get_saturation_pressure_data()

    
    @staticmethod
    def _parse_components_from_comment(filepath: str) -> list[str]:

        components = []

        # ^# matches a line starting with #
        # \s* matches any number of spaces
        # (component | chemical) matches either word 
        pattern = re.compile(r"^#\s*(component|chemical)\b[\s:]*(.*)", re.IGNORECASE)

        with open(filepath, "r") as f:
            for line in f:
                if pattern.match(line):
                    comment_line = line.strip()
                    components.append(comment_line.split()[-1])
        
        # checking that exactly two components are provided
        if len(components) != 2:
            raise VLEDataError(" Binary interaction parameters regression requires exactly two components. ")
        
        return components
    

    @staticmethod
    def _parse_source_from_comment(filepath: str) -> str:
        
        source = "no data source provided"

        # ^#\s* matches '#' followed by optional spaces
        # (source|reference) \b matches the word 'source' or 'reference' followed by a word boundary
        # [\s:]* ignores any spaces or colons immediately following the word
        # (.*) captures the rest of the line (the actual source text)
        pattern = re.compile(r"^#\s*(source|reference)\b[\s:]*(.*)", re.IGNORECASE)

        with open(filepath, "r") as f:
            for line in f:
                if pattern.match(line):
                    source = pattern.match(line).group(2).strip() 
                    break
        return source


    def _extract_data_from_data_frame(self,
                                      dataframe: pd.DataFrame) -> RawExperimentalData:

        " Function to extract process condition data and composition data from the dataframe. "

        # exctracting process condition data
        pressure_Pa_data   = dataframe["P (atm)"].to_numpy() * 1e5    # converting from bar to Pa
        temperature_K_data = dataframe["T (degC)"].to_numpy() + 273.15
        
        # extracting componens fraction data
        for comp in self.components:
            col_x = f"x_{comp}"
            col_y = f"y_{comp}"
            if col_x in dataframe.columns and col_y in dataframe.columns:
                x1_data = dataframe[col_x].to_numpy()
                y1_data = dataframe[col_y].to_numpy()
                break
        else:
            raise VLEDataError("No matching x_* / y_* columns found for given components")
        
        raw_data = []
        for k in range(len(pressure_Pa_data)):
            data_point = DataPoint(
                pressure_Pa = pressure_Pa_data[k],
                temperature_K = temperature_K_data[k],
                x1_mol_frac = x1_data[k],
                y1_mol_frac = y1_data[k],
                x2_mol_frac = 1.0 - x1_data[k],
                y2_mol_frac = 1.0 - y1_data[k],
                comp1 = self.components[0],
                comp2 = self.components[1]
            )
            raw_data.append(data_point)    
        
        return RawExperimentalData(
            data_points = raw_data,
            remove_extreme_concentrations=self.remove_extreme_concentrations
        )
    

    def _get_saturation_pressure_data(self) -> None:
    
        """ 
        The method's objectives are twofold: 
        1) get saturation pressures (in Pa) for each component at each value of the temperature in the avaialble VLE data set
        using CoolProp library 
        2) filter out VLE data points at temperatures for which saturation pressure could not have been evaluated 
        (typically when VLE temperature point is above component's critical temperature)
        """

        T_x_y_points_to_remove = []

        for T_x_y_point in self.T_x_y_points: 

            temperature_K = T_x_y_point.temperature_K

            saturation_pressure_Pa_data = []
            for component in self.components:
                saturation_pressure_Pa = self._get_saturation_pressure_value(
                    component_name=component,
                    temperature_K=temperature_K
                )
                saturation_pressure_Pa_data.append(saturation_pressure_Pa)

            if any(np.isnan(saturation_pressure_Pa_data)):
                T_x_y_points_to_remove.append(T_x_y_point)
            else: 
                T_x_y_point.comp_1_saturation_pressure_Pa = saturation_pressure_Pa_data[0]
                T_x_y_point.comp_2_saturation_pressure_Pa = saturation_pressure_Pa_data[1]

        for T_x_y_point in T_x_y_points_to_remove:
            self.T_x_y_points.remove(T_x_y_point)


    def _get_saturation_pressure_value(self, 
                                       component_name: str,
                                       temperature_K: float) -> float:
        
        try: 
            component = Chemical(CAS_from_any(component_name), T=temperature_K*1000)
            saturation_pressure_Pa = component.Psat
            return saturation_pressure_Pa
        except ValueError: 
            msg = (f"Data import [warning]: "
                   f" Saturation pressure calculation failed for T = {temperature_K:.2f} K and "
                   f" component {component_name} for Thermo library. Check if critical temperature" 
                   f" is exceeded or if component is not supported by Thermo. "
                   f" Skipping this data point.")
            print(colored(msg, 'yellow'))
            return float('nan')
