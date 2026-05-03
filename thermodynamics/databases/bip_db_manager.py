import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json 
from datetime import datetime

from chemicals import CAS_from_any

from thermodynamics.activity_models_regression.regression_aux import BinaryInteractionParametersRegression
from thermodynamics.activity_models_regression.optimization import PolynomialExponentialDIPPR, PolynomialNRTL
from thermodynamics.core.properties import WilsonActivityModel, NRTLActivityModel

from termcolor import colored  # for colored text output


class OverwriteAttemptError(Exception):
    pass


class PolynomialException(Exception):
    pass


class BIPDatabaseManager: 

    def __init__(self,
                 database_filepath: str = 'thermodynamics/databases/BIP_database.json'):
        
        self.database_filepath = database_filepath
        self.data = self._load_database()
        
        pass


    def add_entry(self,
                  BIP_estimator:BinaryInteractionParametersRegression,
                  force_overwrite: bool = False) -> None:
    
    
        metadata = self._extract_metadata(
            BIP_estimator=BIP_estimator
        )

        pair_key_data = self._get_component_pair_key(
            component1=BIP_estimator.VLE_data.components[0], 
            component2=BIP_estimator.VLE_data.components[1]
        )
        name_key = pair_key_data['components']
        CAS_key = pair_key_data['CAS_numbers']

        BIP_data = self._extract_BIP_data(
            BIP_estimator=BIP_estimator,
            name_key=name_key
        )

        self._check_overwrite(
            activity_model=metadata['activity_model'],
            force_overwrite=force_overwrite,
            name_key=name_key,
            CAS_key=CAS_key
        )

        self._run_export_checks(
            BIP_estimator=BIP_estimator
        )

        self._insert_entry(
            name_key=name_key,
            CAS_key=CAS_key,
            metadata=metadata,
            BIP_data=BIP_data
        )

        self._save_database()

        pass


    def _load_database(self) -> dict:

        if not os.path.exists(self.database_filepath):
            with open(self.database_filepath, 'w') as f:
                json.dump({}, f)
                return {}

        with open(self.database_filepath, 'r') as f:
            return json.load(f)



    def _get_component_pair_key(self,
                                component1: str, 
                                component2: str) -> dict:

        " Method to sort component names in alphabetical order to avoid confusion between (A, B) "
        " and (B, A) component pairs. This method also retrieves components CAS numbers to ensure "
        " uniqueness. "
    
        names_sorted = sorted([component1, component2])

        CAS_sorted = []
        for name in names_sorted:
            CAS_entry = CAS_from_any(name)
            if CAS_entry is None:
                raise ValueError(f"Could not retrieve CAS number for component: {name}. "
                                 f"Please check the component name.")
            CAS_sorted.append(CAS_entry)

        return {
            'components': '-'.join(names_sorted),
            'CAS_numbers': '-'.join(CAS_sorted)
        }
    

    def _extract_metadata(self,
                          BIP_estimator: BinaryInteractionParametersRegression) -> dict:
        
        component1, component2 = BIP_estimator.VLE_data.components
        VLE_data_source = BIP_estimator.VLE_data.source
        temperature_lb_K = min(
            [point.temperature_K for point in BIP_estimator.VLE_data.raw_data.data_points]
        )
        temperature_ub_K = max(
            [point.temperature_K for point in BIP_estimator.VLE_data.raw_data.data_points]
        )
        pressure_lb_Pa = min(
            [point.pressure_Pa for point in BIP_estimator.VLE_data.raw_data.data_points]
        )
        pressure_ub_Pa = max(
            [point.pressure_Pa for point in BIP_estimator.VLE_data.raw_data.data_points]
        )
        
        activity_model = BIP_estimator.activity_model_backend.__class__.__bases__[0].__name__
        eos_backend = BIP_estimator.eos_backend.__class__.__name__
        polynomial = BIP_estimator.polynomial.polynomial.__class__.__name__
        goodness_of_fit = BIP_estimator.goodness_of_fit

        if BIP_estimator.elementwise_opt_results is not None: 
            regression_method = 'elementwise_regression'
        else: 
            regression_method = 'direct_regression_from_VLE_data'

        return {
            'component1_name': component1,
            'component2_name': component2,
            'component1_CAS': CAS_from_any(component1),
            'component2_CAS': CAS_from_any(component2),
            'activity_model': activity_model,
            'eos_backend': eos_backend,
            'polynomial': polynomial,
            'goodness_of_fit': goodness_of_fit,
            'regression_method': regression_method,
            'VLE_data_source': VLE_data_source,
            'temperature_lb_K': temperature_lb_K,
            'temperature_ub_K': temperature_ub_K,
            'pressure_lb_Pa': pressure_lb_Pa,
            'pressure_ub_Pa': pressure_ub_Pa,
            'date_of_entry_creation': datetime.now().isoformat()
        }      
    

    def _extract_BIP_data(self,
                          BIP_estimator: BinaryInteractionParametersRegression,
                          name_key: str) -> dict:
        
        BIP_names = BIP_estimator.activity_model_backend.get_BIP_names()
        BIP_coeffs = BIP_estimator.BIP_polynomial_coeffs

        # if alpha is fixed, there are 2 coefficients instead of 4, so BIP_coeffs should be adjusted
        if (isinstance(BIP_estimator.activity_model_backend, NRTLActivityModel) and 
            BIP_estimator.activity_model_backend.alpha_is_fixed):
            BIP_coeffs.extend([BIP_estimator.activity_model_backend.alpha] * 2)

        component1, component2 = BIP_estimator.VLE_data.components
        components_swapped = (
            name_key != '-'.join([component1, component2])
        )

        if components_swapped:
            swapped_coeffs = []
            for k in range(0, len(BIP_names), 2):
                swapped_coeffs.extend([BIP_coeffs[k+1], BIP_coeffs[k]])

            BIP_coeffs = swapped_coeffs
        
        BIP_data = {}
        for BIP_name, BIP_coeff in zip(BIP_names, BIP_coeffs):
            BIP_data[BIP_name] = list(BIP_coeff) if hasattr(BIP_coeff, 'shape') else BIP_coeff

        return BIP_data
    


    def _run_export_checks(self,
                           BIP_estimator: BinaryInteractionParametersRegression) -> None:
        
        self._check_polynomial_compatibility(BIP_estimator=BIP_estimator)
        self._check_model_compatibility(BIP_estimator=BIP_estimator)

        pass 


    def _check_polynomial_compatibility(self,
                                        BIP_estimator: BinaryInteractionParametersRegression) -> None:
        
        polynomial = BIP_estimator.polynomial.polynomial
        
        if (isinstance(BIP_estimator.activity_model_backend, WilsonActivityModel) and
            not isinstance(polynomial, PolynomialExponentialDIPPR)
            ):
            raise PolynomialException(
                f" The specified polynomial {polynomial} is not compatible "
                f"with Wilson activity model regression. \n"
                f" Please use {PolynomialExponentialDIPPR} polynomial for regression of "
                f"Wilson BIP parameters. "
            )
        
        if (isinstance(BIP_estimator.activity_model_backend, NRTLActivityModel) and
            not isinstance(polynomial, PolynomialNRTL)
            ):
            raise PolynomialException(
                f" The specified polynomial {polynomial} is not compatible "
                f"with NRTL activity model regression. \n"
                f" Please use {PolynomialNRTL} polynomial for regression of "
                f"NRTL BIP parameters. "
            )
        
    
    def _check_model_compatibility(self,
                                   BIP_estimator: BinaryInteractionParametersRegression) -> None:
        
        activity_model = BIP_estimator.activity_model_backend.__class__.__bases__[0].__name__
        if activity_model not in self.data.keys():
            raise ValueError(
                f" The specified activity model {activity_model} is not compatible with the existing"
                f" database structure. \n"
            )


    def _check_overwrite(self,
                         activity_model: str,
                         force_overwrite: bool,
                         name_key: str,
                         CAS_key: str) -> None:

        overwrite_attempt = False

        if name_key in self.data[activity_model]:
            if not force_overwrite:
                overwrite_attempt = True
                if not self._offer_overwrite(activity_model, name_key, CAS_key):
                    raise OverwriteAttemptError(
                        f" An entry for the component pair {name_key} already exists under the"
                        f" activity model {activity_model}." 
                    )        
            
        if (not overwrite_attempt and 
            CAS_key in self.data[activity_model]):
            if not force_overwrite:
                overwrite_attempt = True
                if not self._offer_overwrite(activity_model, name_key, CAS_key):
                    raise OverwriteAttemptError(
                        f" An entry for the component pair with CAS numbers {CAS_key} already exists "
                        f" under the activity model {activity_model}."
                    )

        pass


    def _offer_overwrite(self,
                         activity_model: str,
                         name_key: str,
                         CAS_key: str) -> bool:
        
        force_overwrite_input = input(
            f" An entry for the component pair {name_key} already exists under the"
            f" activity model {activity_model}. \n"
            f" Do you want to overwrite the existing entry? (y/n): "
        )
        if force_overwrite_input.lower() == 'y':
            return True
        else:
            print(colored(
                f" Entry for {name_key} under {activity_model} was not overwritten. ", 'yellow'
            ))
            return False


    def _insert_entry(self,
                      name_key: str,
                      CAS_key: str,
                      metadata: dict,
                      BIP_data: dict) -> None: 
        
        entry = {
            'metadata': metadata,
            'BIP_data': BIP_data,
        }

        self.data[metadata['activity_model']][name_key] = entry
        self.data[metadata['activity_model']][CAS_key] = entry

        pass


    def _save_database(self) -> None:

        with open(self.database_filepath, 'w') as f:
            json.dump(self.data, f, indent=4)

        print(colored(f" Database successfully updated and saved to {self.database_filepath}.", 'green'))
    

    def get_entry_by_name(self):


        pass


    def get_entry_by_CAS(self):


        pass 


    pass 

    