
from regression_aux import BinaryInteractionParametersRegression
from regression_aux import WilsonActivityModel

if __name__ == '__main__': 

    wilson_BIP_estimator = BinaryInteractionParametersRegression(activity_model='wilson')
    wilson_BIP_estimator.data_import(filepath = 'thermodynamics/thermo_data/VLE_H2O_NH3.csv')
    wilson_BIP_estimator.regress_BIP_parameters()

    pass