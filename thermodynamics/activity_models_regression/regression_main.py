import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from thermodynamics.core.properties import SoaveRedlichKwongEoSBackend

from thermodynamics.activity_models_regression.data_handling import VLEData
from thermodynamics.activity_models_regression.regression_aux import BinaryInteractionParametersRegression
from thermodynamics.activity_models_regression.activity_models_aux import WilsonActivityModelRegression, NRTLActivityModelRegression
from thermodynamics.activity_models_regression.optimization import PolynomialExponentialDIPPR, PolynomialRegular, PolynomialNRTL
from thermodynamics.activity_models_regression.optimization import AbsoluteForm, NormalizedForm

from thermodynamics.databases.bip_db_manager import BIPDatabaseManager


def main():

    VLE_data = VLEData(
        filepath = 'thermodynamics/activity_models_regression/thermo_data/' \
        'VLE_H2O_NH3.csv'
    )
    
    BIP_estimator = BinaryInteractionParametersRegression(
        activity_model_regression=NRTLActivityModelRegression,
        equation_of_state=SoaveRedlichKwongEoSBackend,
        VLE_data=VLE_data,
        polynomial=PolynomialNRTL(degree=4),
        polynomial_form=AbsoluteForm
    )

    BIP_estimator.regress_BIP_parameters_elementwise()
    BIP_estimator.estimate_polynomial_from_elementwise_optimisation()
    BIP_estimator.results_visualization(get_parity_plot=True,
                                        get_VLE_curve=True)
    
    continue_optimization = input(
        " Do you want to continue optimization using the memetic algorithm? (y/n): "
    )
    if continue_optimization.lower() == 'y':
        BIP_estimator.estimate_polynomial_from_VLE_data(n_jobs=4, is_memetic=True, verbose=True)
        BIP_estimator.results_visualization(get_parity_plot=True,
                                            get_VLE_curve=True)


    save_resuls = input(" Do you want to save the regression results to the database? (y/n): ")
    if save_resuls.lower() == 'y':
        bip_database_manager = BIPDatabaseManager()
        bip_database_manager.add_entry(
            BIP_estimator=BIP_estimator
        )


if __name__ == "__main__": 
    main()


