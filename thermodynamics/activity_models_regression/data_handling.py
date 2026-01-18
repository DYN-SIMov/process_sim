import pandas as pd
import numpy as np

from termcolor import colored                             # for colored text output
from dataclasses import dataclass


class DataPoint(): 

    def __init__(self,
                 pressure_Pa: float,
                 temperature_K: float,
                 x1_mol_frac: float,
                 y1_mol_frac: float,
                 x2_mol_frac: float,
                 y2_mol_frac: float,
                 comp1: str,
                 comp2: str):
        self.pressure_Pa = pressure_Pa
        self.temperature_K = temperature_K
        self.x1_mol_frac = x1_mol_frac
        self.y1_mol_frac = y1_mol_frac
        self.x2_mol_frac = x2_mol_frac
        self.y2_mol_frac = y2_mol_frac
        self.comp1 = comp1
        self.comp2 = comp2


class TxyPoint(): 
    
    def __init__(self,
                 data_points: list[DataPoint]):
        self.data = data_points

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
            raise ValueError(" No data points available to search for constant temperature points. ")
 
        reference_temperature_data = [self.data_points[0].temperature_K]
        for point in self.data_points:
            detected_temp_diff = np.abs(np.array(reference_temperature_data) - point.temperature_K)
            if np.all(detected_temp_diff > temperature_K_tol):
                T_x_y_points.append(TxyPoint(data_points = constant_temperature_points))
                constant_temperature_points = []
                reference_temperature_data.append(point.temperature_K)
            else:
                constant_temperature_points.append(point)
                
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
                 filepath: str):
        self.filepath = filepath
        self.components: list[str] = self._parse_components_from_comment(filepath = filepath)
        self.raw_dataframe: pd.DataFrame = pd.read_csv(filepath_or_buffer = filepath, comment='#')
        self.raw_data: RawExperimentalData = self._extract_data(dataframe = self.raw_dataframe)
        self.T_x_y_points: list[DataPoint] = self._detect_T_x_y_data()

    
    @staticmethod
    def _parse_components_from_comment(filepath: str) -> list[str]:

        "Function to parse component names from comment lines in the data file. "

        components = []
        with open(filepath, "r") as f:
            for line in f:
                if line.startswith("# Component"):
                    comment_line = line.strip()
                    components.append(comment_line.split()[-1])
        
        # checking that exactly two components are provided
        if len(components) != 2:
            raise Exception(" Binary interaction parameters regression requires exactly two components. ")
        
        return components
    

    def _extract_data(self,
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
            raise KeyError("No matching x_* / y_* columns found for given components")
        
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
        
        return RawExperimentalData(data_points = raw_data)
    

    def _detect_T_x_y_data(self) -> list[TxyPoint]:

        """
        Detects if the VLE data are isobaric (T-x-y), isothermal (P-x-y), or mixed.
        If mixed, splits into isobaric and isothermal sets.
        """

        T_x_y_data = self.raw_data.find_constant_temperature_points()

        

        pass




    def _detect_data(self, 
                     data:tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        
        """
        Detects if the VLE data are isobaric (T-x-y), isothermal (P-x-y), or mixed.
        If mixed, splits into isobaric and isothermal sets.
        Stores result in self.data_sets.
        """

        self.data_set = []
        pressure_Pa, temperature_K, x_data, y_data = data
        tol = 1e-3  # tolerance for "constant" (Pa or K)

        # filtering out too low and too high concnetrations to avoid singularities during the datafit
        valid_mask = (
            (x_data >= tol) & (x_data <= 1-tol) &
            (y_data >= tol) & (y_data <= 1-tol)
        )
        pressure_Pa = pressure_Pa[valid_mask]
        temperature_K = temperature_K[valid_mask]
        x_data = x_data[valid_mask]
        y_data = y_data[valid_mask]

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


    def data_import(self, filepath: str) -> None:

        if filepath is None: 
            raise ValueError(" No filepath specified for data import. ")

        df = pd.read_csv(filepath_or_buffer = filepath, comment='#')

        components = self._parse_components_from_comment(filepath = filepath)
        self.components = components

        data = self._extract_data(dataframe = df)

        self._detect_data(data = data)







