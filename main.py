import numpy as np 
from core import Stream
from thermodynamics.core.properties import GammaPhiPackage
from thermodynamics.core.properties import SoaveRedlichKwongEoSBackend, WilsonActivityModel


if __name__ == "__main__":


    property_package = GammaPhiPackage(eos_backend = SoaveRedlichKwongEoSBackend, 
                                       activity_model_backend = WilsonActivityModel,
                                       components = ('H2', 'N2', 'NH3', 'CH3OH','C2H5OH'))
    
    feed = Stream(pressure_Pa = 1e5,
                  temperature_K = 300.0,
                  molar_composition = np.array([0.2, 0.3, 0.0, 0.5, 0.0]),
                  molar_flow_mol_s = 1.0,
                  property_package_backend = property_package)
    
    print(feed.vapour_fraction())

    



