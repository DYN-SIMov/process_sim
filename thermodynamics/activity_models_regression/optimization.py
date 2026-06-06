"""
Module to store all classess and functions related to configuring optimization algorithms
"""
import numpy as np

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds as ScipyBounds

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pymoo.optimize import minimize as pymoo_minimize
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed

from typing import Protocol
from enum import Enum

from pymoo.operators.sampling.rnd import FloatRandomSampling


class LocalOptimizationMethod(Enum):
    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"
    CG = "CG"
    BFGS = "BFGS"
    SLSQP = "SLSQP"


class ObjectiveNumericalError(RuntimeError):
    """Raised when the objective hits invalid/overflow/divide-by-zero."""
    pass

class LocalOptimizationError(RuntimeError):
    """Raised when the local optimization fails to converge."""
    pass


class PolynomialInterface(Protocol): 
    """ Protocol for polynomial forms to be used in the regression of BIP parameters. """
    
    def __init__(self,
                 degree: int):
        self.degree = degree
        self.equation_str = None

    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        pass

    def get_bounds_scipy(self) -> tuple:
        pass

    def initial_guess_scipy(self) -> np.ndarray:
        pass



class PolynominalFormInterface(Protocol):
    """ Protocol for the form of the polynomial to be used in the regression of BIP parameters."""

    def __init__(self,
                 polynomial: PolynomialInterface):
        pass

    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        pass

    def get_bounds_scipy(self) -> tuple:
        pass

    def get_initial_guess_scipy(self) -> np.ndarray:
        pass

    def get_absolute_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        pass


class AbsoluteForm(PolynominalFormInterface):
    """ Wrapper for polynomial object - absolute form of polynomial coefficients: """

    def __init__(self,
                 polynomial: PolynomialInterface):
        self.polynomial = polynomial
        self.degree = polynomial.degree
        self.initial_guess = self.polynomial.get_initial_guess_scipy()
        self.scipy_bounds = self.polynomial.get_bounds_scipy()
        self._xl_absolute = [bound[0] for bound in self.scipy_bounds]
        self._xu_absolute = [bound[1] for bound in self.scipy_bounds]
        
        pass


    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        return self.polynomial.evaluate(temperature_K=temperature_K, coeffs=coeffs)


    def get_bounds_scipy(self) -> tuple:
        return self.scipy_bounds


    def get_initial_guess_scipy(self) -> np.ndarray:
        return self.initial_guess
    
    def get_absolute_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        return coeffs

    


class NormalizedForm(PolynominalFormInterface):
    """Wraper for polynomial object - normalized form of polynomial coefficients:"""

    def __init__(self,
                 polynomial: PolynomialInterface):
        self.polynomial = polynomial
        self.degree = polynomial.degree

        scipy_bounds_absolute = self.polynomial.get_bounds_scipy()
        self._xl_absolute = [bound[0] for bound in scipy_bounds_absolute]
        self._xu_absolute = [bound[1] for bound in scipy_bounds_absolute]

        self.initial_guess = self._normalize_coeffs(self.polynomial.get_initial_guess_scipy())
        
        bounds = self._normalize_coeffs(np.array(self.polynomial.get_bounds_scipy()))
        self.scipy_bounds = tuple(
            (bounds[k][0], bounds[k][1]) for k in range(len(bounds))
        )
        
        pass 


    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        
        denormalized_coeffs = self._denormalize_coeffs(coeffs)
        return self.polynomial.evaluate(temperature_K=temperature_K, coeffs=denormalized_coeffs) 


    def get_bounds_scipy(self) -> tuple:
        return self.scipy_bounds
    

    def get_initial_guess_scipy(self) -> np.ndarray:
        return self.initial_guess
    

    def get_absolute_coeffs(self, coeffs):
        return self._denormalize_coeffs(coeffs)


    def _normalize_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        
        normalized_coeffs = []
        for k in range(len(coeffs)):
            coeff_k = coeffs[k]
            xl_k = self._xl_absolute[k]
            xu_k = self._xu_absolute[k]
            normalized_coeff_k = (coeff_k - xl_k) / (xu_k - xl_k)
            normalized_coeffs.append(normalized_coeff_k)

        return np.array(normalized_coeffs)


    def _denormalize_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        denormalized_coeffs = []
        for k in range(len(coeffs)):
            coeff_k = coeffs[k]
            xl_k = self._xl_absolute[k]
            xu_k = self._xu_absolute[k]
            denormalized_coeff_k = coeff_k * (xu_k - xl_k) + xl_k
            denormalized_coeffs.append(denormalized_coeff_k)

        return np.array(denormalized_coeffs)



class PolynomialExponentialDIPPR(PolynomialInterface): 

    " 4 parameter DIPPR-style polynomial for BIP: "
    " Lambda_ij = exp(A + B/T + C*ln(T) + D*T). "
    " Based on simplified version of the polynomial form described in docs "
    " of thermo python library (thermo.Wilson). "

    def __init__(self,
                 degree: int):
        self.degree = degree
        self.equation_str = 'BIP = exp(A + B/T + C*ln(T) + D*T)'
        pass


    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        """
        Method to evaluate the DIPPR polynomial with exponential term
        """

        T = temperature_K
        A,B,C,D = coeffs

        ln_lambda = (A + 
                     B/T + 
                     C*np.log(T) + 
                     D*T)

        return np.exp(ln_lambda)
    

    def get_bounds_scipy(self) -> tuple:
        """
        Method to get bounds for the DIPPR polynomial coefficients for scipy optimization
        """

        bounds = ((-1e4,1e4),   # A
                  (-1e4,1e4),   # B
                  (-1e3,1e3),   # C
                  (-1e1,1e1))   # D

        return bounds
    

    def get_initial_guess_scipy(self) -> np.ndarray:
        """
        Method to get initial guess for the DIPPR polynomial coefficients for scipy optimization
        """

        initial_guess = np.array([0.5, 0.1, 0.1, 0.1])

        return initial_guess
    


class PolynomialNRTL(PolynomialInterface): 

    " 4 parameter polynominal for NRTL model: "
    " Lambda_ij = exp(A + B/T + C*ln(T) + D*T). "
    " Based on simplified version of the polynomial form described in docs "
    " of thermo python library (thermo.Wilson). "

    def __init__(self,
                 degree: int):
        self.degree = degree
        self.equation_str = 'BIP = A + B/T + C*ln(T) + D*T'
        pass


    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        """
        Method to evaluate the DIPPR polynomial with exponential term
        """

        T = temperature_K
        A,B,C,D = coeffs

        tau = A + B/T + C*np.log(T) + D*T

        return tau
    

    def get_bounds_scipy(self) -> tuple:
        """
        Method to get bounds for the DIPPR polynomial coefficients for scipy optimization
        """

        bounds = ((-1e2,1e2),   # A
                  (-1e3,1e3),   # B
                  (-1e3,1e3),   # C
                  (-1e1,1e1))   # D

        return bounds
    

    def get_initial_guess_scipy(self) -> np.ndarray:
        """
        Method to get initial guess for the DIPPR polynomial coefficients for scipy optimization
        """

        initial_guess = np.array([0.5, 0.1, 0.1, 0.1])

        return initial_guess




class PolynomialRegular(PolynomialInterface): 
    
    " Regular polynomial for BIP: "
    " Lambda_ij = A + B*T + C*T^2 + D*T^3. "

    def __init__(self,
                 degree: int):
        self.degree = degree
        self.equation_str = 'BIP = A + B*T + C*T^2 + D*T^3'
        pass


    def evaluate(self,
                 temperature_K: np.ndarray,
                 coeffs: np.ndarray) -> np.ndarray:
        """
        Method to evaluate the regular polynomial
        """

        T = temperature_K
        A,B,C,D = coeffs

        lambda_ij = (A + 
                     B*T + 
                     C*T**2 + 
                     D*T**3)

        return lambda_ij
    

    def get_bounds_scipy(self) -> tuple:
        """
        Method to get bounds for the regular polynomial coefficients for scipy optimization
        """

        bounds = ((-1e4,1e4),   # A
                  (-1e3,1e3),   # B
                  (-1e3,1e3),   # C
                  (-1e2,1e2))   # D

        return bounds
    

    def get_initial_guess_scipy(self) -> np.ndarray:
        """
        Method to get initial guess for the regular polynomial coefficients for scipy optimization
        """

        initial_guess = np.array([0.5, 0.1, 0.1, 0.1])

        return initial_guess
    


class PolynomialElementwiseEstimator():

    def __init__(self,
                 temperature_K_data: np.ndarray,
                 BIP_elementwise_results: np.ndarray,
                 polynomial:PolynomialInterface):
        self.temperature_K_data = temperature_K_data
        self.BIP_elementwise_results = BIP_elementwise_results
        self.polynomial = polynomial
        pass


    def _objective_function(self,
                            coeffs: np.ndarray,
                            current_BIP_count: int) -> float:
        
        " Objective function for DIPPR polynomial regression of Wilson BIP parameters. "

        " implementation is safeguarded for any potential errors that can be thrown by numpy"

        try:
            with np.errstate(invalid="raise", divide="raise", over="raise"):
                error_data = []
                for k in range(len(self.temperature_K_data)):
                    BIP_ij_exp  = self.BIP_elementwise_results[k, current_BIP_count]
                    BIP_ij_calc = self.polynomial.evaluate(
                        temperature_K = self.temperature_K_data[k],
                        coeffs = coeffs
                    )
                    error = (BIP_ij_exp - BIP_ij_calc)**2
                    error_data.append(error)

                return sum(error_data)

        except FloatingPointError as e:
            return 1e20


    def estimate_coefficients(self) -> list:

        BIP_coeffs_estimation_results = []

        total_number_of_BIP = self.BIP_elementwise_results.shape[1]

        for current_BIP_count in range(0,total_number_of_BIP):

            current_esitmation_run = scipy_minimize(
                fun = self._objective_function,
                    x0 = self.polynomial.get_initial_guess_scipy(),
                    method = LocalOptimizationMethod.NELDER_MEAD.value,
                    args = (current_BIP_count,),
                    bounds = self.polynomial.get_bounds_scipy()
                )

            if not self._optimization_converged(current_esitmation_run):
                raise LocalOptimizationError(
                    f" BIP estimation failed for BIP number {current_BIP_count}. "
                    f"Optimizer output message: {current_esitmation_run.message}"
                )

            BIP_coeffs_estimation_results.append(current_esitmation_run)

        return BIP_coeffs_estimation_results
    

    @staticmethod
    def _optimization_converged(opt_result) -> bool: 

        condition = (
            opt_result.success or
            opt_result.message in [
                    "Optimization terminated successfully.", 
                    "Maximum number of iterations has been exceeded.",
                    "Maximum number of function evaluations has been exceeded."
            ]
        ) 
        return condition




class PymooPolynomialEstimator(Problem):
    
    def __init__(self, 
                 parameter_mapper, 
                 objective_function,
                 VLE_data,
                 polynomial,
                 n_jobs = 1,
                 backend = 'loky', 
                 **kwargs):
        
        self.parameter_mapper = parameter_mapper
        self.objective_function = objective_function
        self.VLE_data = VLE_data
        self.polynomial = polynomial
        self.n_jobs = n_jobs
        self.backend = backend

        config = self.parameter_mapper.get_pymoo_config()
        number_of_parameters = config['number_of_parameters']
        lower_bounds = config['lower_bounds']
        upper_bounds = config['upper_bounds']

        super().__init__(
            n_var=number_of_parameters,
            n_obj=1,
            n_ieq_constr=0,
            xl=lower_bounds,
            xu=upper_bounds,
            **kwargs,
        )


    def _evaluate_with_warnings(self, coeffs:np.ndarray): 

        """
        Runs objective function catching numpy warnins. 

        This is needed because sometimes during optimization some coeffs values
        may lead to invalid calculations (e.g. infiinty) inside the activity model
        objective function.
        """

        try: 
            with np.errstate(invalid="raise", divide="raise", over="raise"):

                objective_function_val = self.objective_function(
                    coeffs=coeffs,
                    VLE_data=self.VLE_data,
                    polynomial=self.polynomial
                )
                return objective_function_val
            
        except FloatingPointError as e: 
            return 1e20


    def _evaluate(self, X, out, *args, **kwargs):

        F = JoblibParallel(n_jobs=self.n_jobs, backend=self.backend)(
            joblib_delayed(self._evaluate_with_warnings)(x) for x in X)

        out["F"] = np.asarray(F, dtype=float).reshape(-1, 1)






class PymooCallbackHandler(Callback):
    def __init__(self,
                 n_gens_skipped: int = 10,
                 n_candidates_selected: int = 3,
                 local_optimizer_method: LocalOptimizationMethod = LocalOptimizationMethod.NELDER_MEAD,
                 local_optimizer_maxiter: int = 100,
                 max_workers: int = 4,
                 verbose: bool = False):
        
        super().__init__()
        self.n_gens_skipped = n_gens_skipped
        self.n_candidates_selected = n_candidates_selected
        self.local_optimizer_method = local_optimizer_method
        self.local_optimizer_maxiter = local_optimizer_maxiter
        self.max_workers = max_workers
        self.verbose = verbose

        pass


    def notify(self, algorithm) -> None:

        """
        Callback function to be implemented for memetic optimization in pymoo.
        """
        current_generation = algorithm.n_iter 
        if current_generation % self.n_gens_skipped != 0:
            return 

        memetic_results = self._run_memetic_optimization(algorithm=algorithm)
        self._update_candidates_in_population(algorithm=algorithm, memetic_results=memetic_results)

        pass


    def _run_memetic_optimization(self, algorithm):

        " Method to run local optimization for the best candidates in the population. "

        
        fvals_data = algorithm.pop.get("F").flatten()
        best_fvals_indices = np.argsort(fvals_data)[:self.n_candidates_selected]
        best_candidates = algorithm.pop[best_fvals_indices]
        theta_initial_data = best_candidates.get("X")
        fval_initial_data = best_candidates.get("F")

        n_jobs = min(
            self.max_workers, 
            self.n_candidates_selected,
            algorithm.problem.n_jobs
        )

        opt_results = JoblibParallel(n_jobs=n_jobs, backend='loky')(
            joblib_delayed(self._run_local_optimization)(
                algorithm=algorithm,
                theta_initial=theta_initial_data[k],
                local_optimizer_method=self.local_optimizer_method,
                local_optimizer_maxiter=self.local_optimizer_maxiter,
            ) for k in range(len(best_candidates))
        )

        return {
            'best_candidates_indices': best_fvals_indices,
            'best_candidates_initial_theta': theta_initial_data,
            'best_candidates_initial_fval': fval_initial_data,
            'opt_results': opt_results
        }


    def _run_local_optimization(self, 
                                algorithm,
                                theta_initial: np.ndarray, 
                                local_optimizer_method: LocalOptimizationMethod, 
                                local_optimizer_maxiter: int) -> dict:

        " Method to run local optimization for a single candidate. "

        xl = algorithm.problem.xl
        xu = algorithm.problem.xu
        bounds = [(xl[k], xu[k]) for k in range(len(xl))]

        try: 
            local_opt_result = scipy_minimize(
                fun=algorithm.problem._evaluate_with_warnings,
                x0=theta_initial,
                method=local_optimizer_method.value,
                bounds=bounds,
                options={'maxiter': local_optimizer_maxiter}
            )
            if not self._local_optimization_converged(local_opt_result):
                raise LocalOptimizationError()
            
            return {
                'theta_opt': local_opt_result.x,
                'fval_opt': local_opt_result.fun
            }

        except Exception as e:
            print(f" Local optimization failed for candidate with initial theta {theta_initial}. "
                  f"Error: {str(e)}")
            return {
                'theta_opt': None,
                'fval_opt': None
            }
    

    def _update_candidates_in_population(self, 
                                         algorithm,
                                         memetic_results: dict) -> None:

        " Method to update the candidate in the population with the optimized values. "

        for idx, candidate_idx in enumerate(memetic_results['best_candidates_indices']):
            theta_opt = memetic_results['opt_results'][idx]['theta_opt']
            fval_opt = memetic_results['opt_results'][idx]['fval_opt']
            theta_initial = memetic_results['best_candidates_initial_theta'][idx]
            fval_initial = memetic_results['best_candidates_initial_fval'][idx][0]

            if theta_opt is None or fval_opt is None:
                continue

            algorithm.pop[candidate_idx].set("X", theta_opt)
            algorithm.pop[candidate_idx].set("F", np.asarray(fval_opt).reshape(1,))

            if self.verbose:
                print(
                    self._get_memetic_optimization_message(
                        theta_initial=theta_initial,
                        fval_initial=fval_initial,
                        theta_opt=theta_opt,
                        fval_opt=fval_opt
                    )
                )
            

    @staticmethod
    def _get_memetic_optimization_message(theta_initial: np.ndarray,
                                          fval_initial: float,
                                          theta_opt: np.ndarray,
                                          fval_opt: float) -> str: 

        " Method to get the message for memetic optimization results. "

        msg = (
            f"Memetic optimization successful: \n"
            f" Initial fval: {fval_initial:.3f}, initial coeffs =     \
                { [round(x, 2) for x in theta_initial.tolist()] }, \n"
            f" Optimized fval: {fval_opt:.3f}, optimized coeffs = \
                { [round(x, 2) for x in theta_opt.tolist()] } \n"
        )

        return msg 


    @staticmethod
    def _local_optimization_converged(local_opt_result) -> bool: 

        condition = (
            local_opt_result.success or
            local_opt_result.message in [
                    "Optimization terminated successfully.", 
                    "Maximum number of iterations has been exceeded."
            ]
        ) 
        return condition
    

class OwnBiasedSampling(FloatRandomSampling):

    def __init__(self,
                 BIP_config: list,
                 fraction_of_biased_samples: float = 0.5,
                 bias_strength: float = 0.5,
                 seed: int = 42):
        super().__init__()
        self.BIP_config = BIP_config
        self.fraction_of_biased_samples = fraction_of_biased_samples
        self.bias_strength = bias_strength
        self.seed = seed

        self.random_engine = np.random.default_rng(seed=self.seed)
        self.initial_guess = self._get_initial_guess()


    def _get_initial_guess(self) -> np.ndarray:

        initial_guess = []
        for bip in self.BIP_config:
            if bip.is_regressed and bip.is_temperature_dependant:
                initial_guess.extend(bip.value)
            elif bip.is_regressed and not bip.is_temperature_dependant:
                initial_guess.append(bip.value)

        return np.array(initial_guess)


    def _do(self, 
            problem, 
            n_samples,
            random_state=None,
            *args,
            **kwargs
        ) -> np.ndarray:    

        n_biased = int(n_samples * self.fraction_of_biased_samples)
        n_random = n_samples - n_biased
        xl = problem.xl
        xu = problem.xu

        random_samples = self.random_engine.uniform(
            low=xl, high=xu, size=(n_random, xl.shape[0])
        )

        biased_samples = (
            self.initial_guess + 
            self.bias_strength * self.random_engine.uniform(
                low=-1.0, high=1.0, size=(n_biased, xl.shape[0])
            )
        )

        if problem.has_bounds():
            biased_samples = np.clip(biased_samples, xl, xu)

        population = np.vstack((random_samples, biased_samples))

        return population
                


class OptimizationVectorMapper(): 

    def __init__(self,
                 activity_model,
                 polynomial):
        self.activity_model = activity_model
        self.polynomial = polynomial

        self.temp_dep_bips = self.activity_model.get_temp_dependant_BIPs()
        self.temp_indep_bips = self.activity_model.get_temp_independant_BIPs()

        pass


    def elementwise_regression_adapter(self, optimization_vector: np.ndarray) -> np.ndarray:

        if len(optimization_vector) == len(self.temp_dep_bips) + len(self.temp_indep_bips):
            return optimization_vector
        
        for bip in self.temp_indep_bips:
            optimization_vector = np.append(optimization_vector, bip.initial_guess)

        return optimization_vector


    def get_pymoo_config(self) -> dict: 

        lower_bounds = []
        upper_bounds = []

        for bip in self.activity_model.BIPs: 
            if not bip.is_regressed:
                continue

            if bip.is_temperature_dependant:
                bounds_tuple = self.polynomial.get_bounds_scipy()
                lower_bounds.extend([bounds_tuple[k][0] for k in range(len(bounds_tuple))])
                upper_bounds.extend([bounds_tuple[k][1] for k in range(len(bounds_tuple))])
            else:
                lower_bounds.append(bip.bounds[0])
                upper_bounds.append(bip.bounds[1])

        number_of_parameters = len(lower_bounds)

        config = {
            'number_of_parameters': number_of_parameters,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds
        }

        return config


    def decode_pymoo_vector(self, optimization_vector) -> list: 

        decoded_map = {}
        param_ind = 0
        degree = self.polynomial.degree

        for bip in self.activity_model.BIPs: 
            if not bip.is_regressed:
                continue

            if bip.is_temperature_dependant: 
                decoded_map[bip.name] = optimization_vector[param_ind : param_ind + degree]
                param_ind += degree
            else:
                decoded_map[bip.name] = optimization_vector[param_ind]
                param_ind += 1

        return decoded_map