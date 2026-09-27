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

from functools import partial

from thermodynamics.parsers.pdf_processing import VLEDataPDFParser

def main():

    VLE_data = VLEData(
        filepath = 'thermodynamics/activity_models_regression/thermo_data/' \
        'VLE_MeOH_H2O.csv'
    )
    
    nrtl_varying_alpha = partial(NRTLActivityModelRegression, alpha_is_fixed=False, alpha=0.2)

    BIP_estimator = BinaryInteractionParametersRegression(
        activity_model_regression=NRTLActivityModelRegression,
        equation_of_state=SoaveRedlichKwongEoSBackend,
        VLE_data=VLE_data,
        polynomial=PolynomialNRTL(degree=4),
        polynomial_form=AbsoluteForm
    )

    BIP_estimator.estimate_polynomial_elementwise()
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
    # main()

    ammonia_path = "/Users/maksimov.ps95/Desktop/Reading/Thermodynamics/VLE data/NH3 - H2O VLE data.pdf"
    methanol_path = "/Users/maksimov.ps95/Desktop/Reading/Thermodynamics/VLE data/MeOH - H2O VLE data.pdf"
    test_own = "/Users/maksimov.ps95/Desktop/Maksimovetal.2022.pdf"

    vle_parser = VLEDataPDFParser()
    vle_parser.get_VLE_data_from_pdf(
        pdf_path=test_own
    )

    pass


"""
TODO: 
1) 


"""
