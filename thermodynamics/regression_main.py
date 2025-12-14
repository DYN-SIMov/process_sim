
from regression_aux import BinaryInteractionParametersRegression

if __name__ == '__main__': 

    wilson_BIP_estimator = BinaryInteractionParametersRegression(activity_model='wilson',
                                                                 equation_of_state='srk',)
    wilson_BIP_estimator.data_import(filepath = 'thermodynamics/thermo_data/VLE_H2O_NH3.csv')
    wilson_BIP_estimator.regress_BIP_parameters_elementwise()
    wilson_BIP_estimator.estimate_DIPPR_polynomial_from_elementwise_optimisation()

    pass