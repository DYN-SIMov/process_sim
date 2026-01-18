"""
Module to store all classess and functions related to configuring optimization algorithms
"""
import numpy as np

from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize as pymoo_minimize
from joblib import Parallel as JoblibParallel
from joblib import delayed as joblib_delayed

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

