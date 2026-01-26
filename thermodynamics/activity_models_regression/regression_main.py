import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from thermodynamics.core.properties import SoaveRedlichKwongEoSBackend

from data_handling import VLEData
from regression_aux import BinaryInteractionParametersRegression
from activity_models_aux import WilsonActivityModelRegression
from optimization import PolynomialExponentialDIPPR



def main():

    VLE_data = VLEData(filepath = 'thermodynamics/activity_models_regression/thermo_data/VLE_isobaric_MeOH_H2O.csv')
    # VLE_data = VLEData(filepath = 'thermodynamics/activity_models_regression/thermo_data/VLE_H2O_NH3.csv')
    
    wilson_BIP_estimator = BinaryInteractionParametersRegression(activity_model_regression=WilsonActivityModelRegression,
                                                                 equation_of_state=SoaveRedlichKwongEoSBackend,
                                                                 VLE_data=VLE_data,
                                                                 polynomial=PolynomialExponentialDIPPR(degree=4))
    wilson_BIP_estimator.regress_BIP_parameters_elementwise()
    wilson_BIP_estimator.estimate_polynomial_from_elementwise_optimisation()
    # wilson_BIP_estimator.estimate_DIPPR_polynomial_from_elementwise_optimisation()
    # wilson_BIP_estimator.estimate_DIPPR_polynomial_from_VLE_data()
    wilson_BIP_estimator.results_visualization(get_parity_plot=True,
                                               get_VLE_curve=True)

    pass


if __name__ == "__main__": 
    main()



" TODO: "
" 1) implement direct estimation of the polynomial coefficients with pymoo "
"   --> see if it improves fit for MeOH-H2O with Wilson"
" 2) add NRTL and see if it works better with the 4p polynomial for MeOH-H2O "
" 3) add simple regression from elementwise BIP's optimisation with numpy.polyfit"
" 4) add the functionality for recording the results of the polynomial's coefficient"