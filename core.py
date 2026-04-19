import numpy as np
from thermodynamics.core.properties import PropertyPackageInterface
from dataclasses import dataclass 

@dataclass
class Stream: 

    """ State container to store properties of streams in between unit ops. """

    pressure_Pa: float                  
    temperature_K: float           
    molar_composition: np.ndarray       # order according to property package backend components
    molar_flow_mol_s: float             
    property_package_backend: PropertyPackageInterface     

    def __post_init__(self):
        # Validate molar composition sums to 1
        if not np.isclose(np.sum(self.molar_composition), 1.0, atol=1e-12):
            raise ValueError("Molar_composition must sum to 1.0")
        
        # Validate no negative entries in composition
        if (self.molar_composition < -1e-12).any():
            raise ValueError("Composition has negative entries.")
        
        # Validate composition length matches backend components
        if len(self.molar_composition) != len(self.property_package_backend.components):
            raise ValueError("Composition length does not match backend components.")

    def vapour_fraction(self) -> float:
        
        flash_result = self.property_package_backend.TP_flash(pressure_Pa = self.pressure_Pa,
                                                              temperature_K = self.temperature_K,
                                                              molar_composition = self.molar_composition)
        

        # Placeholder implementation
        return 0.0

    def compressibility_factor(self) -> float:
        return self.backend.get_compressibility_factor(self.temperature_K, self.pressure_Pa, self.molar_composition)
    
    def density_SI(self) -> float:

        return self.backend.get_density_SI(self.temperature_K, self.pressure_Pa, self.molar_composition)

    