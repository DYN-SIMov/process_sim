
from regression_aux import BinaryInteractionParametersRegression

if __name__ == '__main__': 

    wilson_BIP_estimator = BinaryInteractionParametersRegression(activity_model='wilson',
                                                                 equation_of_state='srk',)
    wilson_BIP_estimator.data_import(filepath = 'thermodynamics/thermo_data/VLE_H2O_NH3.csv')
    wilson_BIP_estimator.regress_BIP_parameters_elementwise()
    # wilson_BIP_estimator.estimate_DIPPR_polynomial_from_elementwise_optimisation()
    wilson_BIP_estimator.estimate_DIPPR_polynomial_from_VLE_data()
    wilson_BIP_estimator.results_visualization(get_parity_plot=True,
                                               get_VLE_curve=True)

    pass


" TODO: "
" 1) implement direct estimation of the polynomial coefficients with pymoo "
"   --> see if it improves fit for MeOH-H2O with Wilson"
" 2) add NRTL and see if it works better with the 4p polynomial for MeOH-H2O "
" 3) add simple regression from elementwise BIP's optimisation with numpy.polyfit"
" 4) add the functionality for recording the results of the polynomial's coefficient"