import json 
import os 
from datetime import datetime

from activity_models_regression.regression_aux import BinaryInteractionParametersRegression

class BIPDatabaseManager: 

    def __init__(self,
                 database_filepath: str = 'thermodynamics/databases/BIP_database.json'):
        
        self.database_filepath = database_filepath
        if not os.path.exists(self.database_filepath):
            with open(self.database_filepath, 'w') as f:
                json.dump({}, f)


        pass


    def add_entry(self,
                  BIP_estimator:BinaryInteractionParametersRegression) -> None:
    


        pass


    def _get_component_pair_key(self,
                                component1: str, 
                                component2: str) -> str:

        " Method to sort component names in alphabetical order to avoid confusion between (A, B) "
        " and (B, A) component pairs. "

        return "-".join(sorted([component1, component2]))


    pass 

    