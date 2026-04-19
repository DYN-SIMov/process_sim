import numpy as np
from typing import Protocol

from chemicals import CAS_from_any
from chemicals import MW as MW_data_import
from chemicals.acentric import omega as acentric_factor_data_import
from chemicals.critical import Tc as critical_temperature_data_import, Pc as critical_pressure_data_import

from CoolProp.CoolProp import PropsSI

from scipy.constants import R


class PureComponentDataBackend():

    """ Pure component property package backend. """
    components: tuple[str, ...]  

    def __init__(self, 
                 components: tuple[str, ...]):
        self.components = components

        self.acentric_factor_data = np.zeros_like(a=components,dtype = float)
        self.critical_temperature_K_data = np.zeros_like(a=components,dtype = float)
        self.critical_pressure_Pa_data = np.zeros_like(a=components,dtype = float)
        for k in range(len(components)):
            compoent_CASRN = CAS_from_any(components[k])
            self.acentric_factor_data[k] = acentric_factor_data_import(CASRN = compoent_CASRN)  
            self.critical_temperature_K_data[k] = critical_temperature_data_import(CASRN = compoent_CASRN)  
            self.critical_pressure_Pa_data[k] = critical_pressure_data_import(CASRN = compoent_CASRN) 



class EquationOfStateInterface(Protocol): 
    
    """ Common interface for equation-of-state backends. """
    name: str
    components: tuple[str, ...]       
    pure_component_data_backend: PureComponentDataBackend
    
    def get_fugacity_coefs(self,
                           temperature_K: float, 
                           pressure_Pa: float, 
                           molar_composition: np.ndarray) -> np.ndarray: ...

    def get_compressibility_factor(self, 
                                   temperature_K: float,
                                   pressure_Pa: float,
                                   molar_composition: np.ndarray) -> float: ...
    
    def get_density_SI(self,
                       temperature_K: float,
                       pressure_Pa: float,
                       molar_composition: np.ndarray) -> float: ...
    


class ActivityModelInterface(Protocol):

    """ Common interface for activity model backends. """
    name: str
    components: tuple[str, ...]       
    
    def get_activity(self, 
                     temperature_K: float,
                     pressure_Pa: float,
                     molar_composition: np.ndarray) -> np.ndarray: ...
    



class PropertyPackageInterface(Protocol):

    """ Common interface for property packages. """
    eos_backend: EquationOfStateInterface
    activity_model_backend: ActivityModelInterface

    def _TPD_stability_test(self): ...
    





class GammaPhiPackage():

    """ Property package to calculate VLE using activity and EOS models. """
    def __init__(self, 
                 eos_backend: EquationOfStateInterface,
                 activity_model_backend: ActivityModelInterface,
                 components: tuple[str, ...]):
        
        pure_component_data_backend = PureComponentDataBackend(components = components)

        self.eos_backend = eos_backend(components = components,
                                       pure_component_data_backend = pure_component_data_backend)
        self.activity_model_backend = activity_model_backend(components = components,
                                                             pure_component_data_backend = pure_component_data_backend)
        self.components = components


    def _TPD_stability_test(self, 
                            temperature_K: float,
                            pressure_Pa: float,
                            molar_composition: np.ndarray) -> bool:
        " Performs a Tangent Plane Stability test for a given conditions (T, P, z). "
        " The method is based on Michelsen (1982)"

        fugacity_coefs: np.ndarray = self.eos_backend.get_fugacity_coefs(temperature_K = temperature_K,
                                                                         pressure_Pa = pressure_Pa,
                                                                         molar_composition = molar_composition)
        
        activity_coefs: np.ndarray = self.activity_model_backend.get_effective_activity_coefs(temperature_K = temperature_K,
                                                                                              pressure_Pa = pressure_Pa,
                                                                                              molar_composition = molar_composition)
        


        return True


    def TP_flash(self, 
                 temperature_K: float,
                 pressure_Pa: float,
                 molar_composition: np.ndarray ) -> None: 

        " Isothermal-isobaric flash calculation. "
        TPD_stability: bool = self._TPD_stability_test(temperature_K = temperature_K,
                                                       pressure_Pa = pressure_Pa,
                                                       molar_composition = molar_composition)


        return None



class IdealGasBackend:

    """ Ideal gas property package backend. """
    name = 'IdealGas'
    
    def __init__(self, 
                 components: tuple[str, ...]):
        self.components = components

    def get_compressibility_factor(self, 
              temperature_K: float, 
              pressure_Pa: float, 
              molar_composition: np.ndarray,) -> float:
        return 1.0
    
    def get_density_SI(self,
                       temperature_K: float,
                       pressure_Pa: float,
                       molar_composition: np.ndarray) -> float:
        
        # placeholder implementation
        return 0.9
    


class SoaveRedlichKwongEoSBackend():
    name = 'SRK'
    components: tuple[str, ...]
    pure_component_data_backend: PureComponentDataBackend

    
    def __init__(self, 
                 components: tuple[str, ...],
                 pure_component_data_backend: PureComponentDataBackend):
        self.components = components
        self.pure_component_data_backend = pure_component_data_backend
        self._cache = {}

    
    def _get_SRK_parameters(self, 
                            temperature_K: float, 
                            pressure_Pa: float, 
                            molar_composition: np.ndarray) -> dict:
        
        " This method calculates SRK EOS parameters (a, b, A, B). "

        cache_key = (temperature_K, pressure_Pa, tuple(molar_composition))
        # Return cached values if available
        if cache_key in self._cache:
            return self._cache[cache_key]

        omega = self.pure_component_data_backend.acentric_factor_data
        Tc = self.pure_component_data_backend.critical_temperature_K_data
        Pc = self.pure_component_data_backend.critical_pressure_Pa_data

        if len(molar_composition) > 1: 
            m   = 0.48 + 1.574 * omega - 0.176 * omega**2
            T_r = temperature_K / Tc
            alpha = (1 + m * (1 - np.sqrt(T_r)))**2       
            
            # a_c = 0.42747 * (R**2 * Tc**2) / Pc
            # b   = 0.08664 * (R * Tc) / Pc

            # a = a_c * alpha 

            # a_mix = (molar_composition @ np.sqrt(a))**2
            # b_mix = molar_composition @ b 

            A_mix = 0.42747 * pressure_Pa / temperature_K**2 * ( molar_composition @ (Tc * np.sqrt(alpha) / np.sqrt(Pc)) )**2
            B_mix = 0.08664 * pressure_Pa / temperature_K * ( molar_composition @ (Tc / Pc) )

        
        params = {
            'alpha': alpha,
            'A_mix': A_mix,
            'B_mix': B_mix,
            'Tc': Tc,
            'Pc': Pc,
        }
        
        # Cache the results
        self._cache = {cache_key: params} 
        return params


    def get_compressibility_factor(self, 
                                   phase: str, 
                                   temperature_K: float, 
                                   pressure_Pa: float, 
                                   molar_composition: np.ndarray) -> float:
        
        " This method solves the SRK cubic equation of state for Z-factor. "
        " The methods is based on original work by Soave (1972). "

        params = self._get_SRK_parameters(temperature_K, pressure_Pa, molar_composition)
        A_mix  = params['A_mix']
        B_mix  = params['B_mix']

        # Cubic equation coefficients: Z^3 - Z^2 + (A - B - B^2)Z - AB = 0
        coefs = [1, -1, A_mix - B_mix - B_mix**2, -1 * A_mix * B_mix]
        roots = np.roots(coefs)

        real_roots = roots[np.abs(roots.imag) < 1e-10].real
        positive_roots = real_roots[real_roots > 0]
        
        if phase == 'v':
            Z_val = np.max(positive_roots) if len(positive_roots) > 0 else None
        elif phase == 'l':
            Z_val = np.min(positive_roots) if len(positive_roots) > 0 else None
        else: 
            raise Exception("Phase is not specified properly, please select either 'v' for vapour or 'l' for liquid")

        if Z_val is None:
            raise ValueError("No physically meaningful compressibility factor found")

        return Z_val
    

    def get_fugacity_coefs(self, 
                           temperature_K: float, 
                           pressure_Pa: float, 
                           molar_composition: np.ndarray) -> np.ndarray:
        
        " This method calculates fugacity coefficients using SRK EOS based on determined Z-factor. "
        " The method is based on original work by Soave (1972). "
        
        params = self._get_SRK_parameters(temperature_K, pressure_Pa, molar_composition)
        alpha  = params['alpha']
        A_mix  = params['A_mix']
        B_mix  = params['B_mix']
        Tc     = params['Tc']
        Pc     = params['Pc']

        Z_val = self.get_compressibility_factor(temperature_K=temperature_K,
                                                pressure_Pa=pressure_Pa,
                                                molar_composition=molar_composition,
                                                phase='v')

        alpha_ratio = np.divide(np.sqrt(alpha) * Tc / np.sqrt(Pc), 
                                molar_composition @ (np.sqrt(alpha) * Tc / np.sqrt(Pc)))
        
        b_ratio = np.divide((Tc / Pc), 
                             molar_composition @ (Tc / Pc))

        phi_log = b_ratio * (Z_val - 1) - np.log(Z_val - B_mix) - A_mix / B_mix * (2 * alpha_ratio - b_ratio) * np.log(1 + B_mix / Z_val)
        
        fugacity_coefs = np.exp(phi_log)
        return fugacity_coefs
        


class WilsonActivityModel():
    name = 'Wilson'
    components: tuple[str, ...]
    pure_component_data_backend: PureComponentDataBackend
    
    def __init__(self,
                 components: tuple[str, ...],
                 pure_component_data_backend: PureComponentDataBackend):
        self.components = components
        self.pure_component_data_backend = pure_component_data_backend
        self._cache = {}
        

    def _components_screening(self,
                             temperature_K,
                             pressure_Pa) -> list[str]:

        " This method screens components for Wilson activity model applicability. "
        " The method returns a list of strings in which each entry corresponds to the law that is applicable for each component"
        " 'HENRY' - Henry's law is applicable"
        " 'RAOULT' - Raoult's law is applicable"

        component_method = []
        for k in range(len(self.components)):

            if temperature_K > self.pure_component_data_backend.critical_temperature_K_data[k]:
                component_method.append('HENRY')

            else:
                try: 
                    # NOTE: fetching saturation pressure from CoolProp relies on correct component name --> query by CAS RN
                    saturation_pressure_Pa = PropsSI('P', 'T', temperature_K, 'Q', 1, CAS_from_any(self.components[k]))
                except Exception: 
                    msg = (
                        f"CoolProp could not retrieve saturation pressure data for component {self.components[k]}"
                        f" at T = {temperature_K} K."
                        " assuming Henry's law is applicable."
                    )
                    print(msg)
                    saturation_pressure_Pa = 0  
                    
                if saturation_pressure_Pa > max(pressure_Pa * 0.01, 1e3):
                    component_method.append('RAOULT')
                else:
                    component_method.append('HENRY')

        return component_method


    def get_effective_activity_coefs(self, 
                           temperature_K: float,
                           pressure_Pa: float,
                           molar_composition: np.ndarray) -> np.ndarray:
        " This method calculates effective activity coefficients using Wilson model. "

        " NOTE: this method returns an array of effective activity coefficients, NOT the activity coefficients themselves. "
        " The effective activity coefficient is defined as follows: "
        " In case of Raoult's law: gamma_i_eff = gamma_i "
        " In case of Henry's law: gamma_i_eff = H_i / P_sat_ref, which is equivalent to activity coefficient at infinite dilution. "

        comp_method = self._components_screening(temperature_K = temperature_K,
                                                pressure_Pa = pressure_Pa)
        

        # Iterate over components and estimate effective activity coefficients based on screening
        for k in range(len(self.components)):
            if comp_method[k] == 'RAOULT':
                # Placeholder implementation
                pass
            elif comp_method[k] == 'HENRY':
                # Placeholder implementation
                pass
            else:
                raise Exception("Component screening method not recognized.")




        # Placeholder implementation
        return np.ones_like(molar_composition)
    


class NRTLActivityModel():
    name = 'NRTL'
    components: tuple[str, ...]
    pure_component_data_backend: PureComponentDataBackend
    
    def __init__(self,
                 components: tuple[str, ...],
                 pure_component_data_backend: PureComponentDataBackend):
        self.components = components
        self.pure_component_data_backend = pure_component_data_backend
        self._cache = {}

    def get_effective_activity_coefs(self,
                                   temperature_K: float,
                                   pressure_Pa: float,
                                   molar_composition: np.ndarray) -> np.ndarray:
        " This method calculates effective activity coefficients using NRTL model. "

        # Placeholder implementation
        return np.ones_like(molar_composition)
    
    
    


