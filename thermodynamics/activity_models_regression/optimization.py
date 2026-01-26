"""
Module to store all classess and functions related to configuring optimization algorithms
"""
import numpy as np

from scipy.optimize import minimize as scipy_minimize

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed



class PolynomialExponentialDIPPR(): 

    " 4 parameter DIPPR-style polynomial for BIP: "
    " Lambda_ij = exp(A + B/T + C*ln(T) + D*T). "
    " Based on simplified version of the polynomial form described in docs "
    " of thermo python library (thermo.Wilson). "

    def __init__(self,
                 degree: int):
        self.degree = degree
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
    


class PolynomialExponentialElementwiseEstimator():

    def __init__(self,
                 temperature_K_data: np.ndarray,
                 BIP_elementwise_results: np.ndarray,
                 polynomial:PolynomialExponentialDIPPR):
        self.temperature_K_data = temperature_K_data
        self.BIP_elementwise_results = BIP_elementwise_results
        self.polynomial = polynomial
        pass


    def _objective_function(self,
                            coeffs: np.ndarray,
                            current_BIP_count: int) -> float:
        
        " Objective function for DIPPR polynomial regression of Wilson BIP parameters. "
        
        error_data = []
        for k in range(len(self.temperature_K_data)):
            BIP_ij_exp  = self.BIP_elementwise_results[k, current_BIP_count]
            BIP_ij_calc = self.polynomial.evaluate(temperature_K = self.temperature_K_data[k],
                                                      coeffs = coeffs)
            error = (BIP_ij_exp - BIP_ij_calc)**2
            error_data.append(error)

        return sum(error_data)


    def estimate_coefficients(self) -> list:

        BIP_coeffs_estimation_results = []

        total_number_of_BIP = self.BIP_elementwise_results.shape[1]

        for current_BIP_count in range(0,total_number_of_BIP):

            current_esitmation_run = scipy_minimize(fun = self._objective_function,
                                        x0 = np.repeat(0.0, self.polynomial.degree),
                                        method = 'Nelder-Mead',
                                        args = (current_BIP_count),
                                        bounds = ((-1e4,1e4),(-1e4,1e4),(-1e3,1e3),(-1e0,1e0)))

            if current_esitmation_run.success:
                BIP_coeffs_estimation_results.append(current_esitmation_run)
            else:
                msg = (f" Polynomial coefficient estimation [error]: "
                       f" Optimization for BIP index {current_BIP_count} did not converge. ")
                raise RuntimeError(msg)

        return BIP_coeffs_estimation_results





class DIPPR_polynomial_regression_GA(Problem):
    def __init__(self, 
                 number_of_parameters,
                 lower_bounds,
                 upper_bounds, 
                 objective_function,
                 n_jobs = 1,
                 backend = 'loky', 
                 **kwargs):
        
        self.objective_function = objective_function
        self.n_jobs = n_jobs
        self.backend = backend

        super().__init__(
            n_var=number_of_parameters,
            n_obj=1,
            n_ieq_constr=0,
            xl=lower_bounds,
            xu=upper_bounds,
            **kwargs,
        )


    def _evaluate_safely(self,coeffs): 

        try: 
            objective_function_val = self.objective_function(coeffs)
        except: 
            objective_function_val = 1e20

        return objective_function_val


    def _evaluate(self, X, out, *args, **kwargs):

        F = JoblibParallel(n_jobs=self.n_jobs, backend=self.backend)(
            joblib_delayed(self._evaluate_safely)(x) for x in X)

        out["F"] = np.asarray(F, dtype=float).reshape(-1, 1)

