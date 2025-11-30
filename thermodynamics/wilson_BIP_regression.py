import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import minimize

from CoolProp.CoolProp import PropsSI
from chemicals import CAS_from_any


df = pd.read_csv(filepath_or_buffer = 'thermodynamics/thermo_data/VLE_isobaric_MeOH_H2O.csv',
                 comment='#')

pressure_Pa = 1e5 
components = ('CH3OH', 'H2O')

data_import        = df.to_numpy()
temperature_K_data = data_import[:,0] + 273.15
x_data             = data_import[:,1]
y_data             = data_import[:,2]


def Wilson_equation(lambda_12: float,
                    lambda_21: float,
                    x_val: np.ndarray) -> np.ndarray:

    " Returns arrays of activity coefficients based on Wilson equations. "

    x_1 = x_val
    x_2 = 1 - x_val

    ln_gamma_1 = (
                    -np.log(x_1 + lambda_12 * x_2) + 
                    x_2 * lambda_12 / (x_1 + lambda_12 * x_2) - 
                    x_2 * lambda_21 / (x_2 + lambda_21 * x_1)
    )

    ln_gamma_2 = (
                    -np.log(x_2 + lambda_21 * x_1) -
                    x_1 * lambda_12 / (x_1 + lambda_12 * x_2) +
                    x_1 * lambda_21 / (x_2 + lambda_21 * x_1)
                )

    gamma_1 = np.exp(ln_gamma_1)
    gamma_2 = np.exp(ln_gamma_2)

    return gamma_1, gamma_2


def objective_function(theta: np.ndarray,
                       y_exp_val: float,
                       x_exp_val: float,
                       saturation_pressure_Pa_1: float,
                       saturation_pressure_Pa_2: float,
                       pressure_Pa: float) -> float:

    " Objective function to minimize for Wilson BIP parameters estimation. "

    lambda_12 = theta[0]
    lambda_21 = theta[1]

    gamma_1_calc, gamma_2_calc = Wilson_equation(lambda_12 = lambda_12,
                                                 lambda_21 = lambda_21,
                                                 x_val = x_exp_val)

    # based on the modified Raoult's law: y_i * P_total= x_i * gamma_i * P_sat_i 
    y_calc_val_1 = x_exp_val * gamma_1_calc * saturation_pressure_Pa_1 / pressure_Pa
    y_calc_val_2 = (1 - x_exp_val) * gamma_2_calc * saturation_pressure_Pa_2 / pressure_Pa

    y_exp_val_1  = y_exp_val
    y_exp_val_2  = 1 - y_exp_val

    error = (y_exp_val_1 - y_calc_val_1)**2 + (y_exp_val_2 - y_calc_val_2)**2

    return error


# elementwise fitting of the Wilson BIP parameters
theta_initial = np.array([1.0, 1.0])

lambda_12_data = []
lambda_21_data = []

for k in range(len(temperature_K_data)):

    saturation_pressure_Pa_1 = PropsSI('P','T',temperature_K_data[k],'Q',1,CAS_from_any(components[0]))
    saturation_pressure_Pa_2 = PropsSI('P','T',temperature_K_data[k],'Q',1,CAS_from_any(components[1]))

    args = (y_data[k],
            x_data[k],
            saturation_pressure_Pa_1,
            saturation_pressure_Pa_2,
            pressure_Pa)

    result = minimize(fun = objective_function,
                      x0 = theta_initial,
                      args = args,
                      method = 'SLSQP',
                      bounds=((1e-3, None), (1e-3, None)))

    msg = (f"Experimental point = {k+1:3}: T = {temperature_K_data[k]:.2f} K, "
           f"x_{components[0]} = {x_data[k]:.3f}, y_{components[0]} = {y_data[k]:.3f} --> "
           f"Fitted Wilson BIP parameters: "
           f" lambda_12 = {result.x[0]:.6f}, "
           f" lambda_21 = {result.x[1]:.6f}. "
           f" residual = {result.fun:.6e}.")
    
    print(msg)

    theta_initial = result.x
    lambda_12_data.append(result.x[0])
    lambda_21_data.append(result.x[1])


# fitting the polynomial correlations to the Wilson BIP parameters data
lambda_12_polynomial_coeffs = np.polyfit(temperature_K_data, lambda_12_data, deg=2)
lambda_21_polynomial_coeffs = np.polyfit(temperature_K_data, lambda_21_data, deg=2)



# visualization of the fitting results
y_calc_MeOH = []
for k in range(len(temperature_K_data)):
    # Use polynomial fit to get lambda values at each temperature
    lambda_12 = np.polyval(lambda_12_polynomial_coeffs, temperature_K_data[k])
    lambda_21 = np.polyval(lambda_21_polynomial_coeffs, temperature_K_data[k])
    gamma_1, gamma_2 = Wilson_equation(lambda_12, lambda_21, x_data[k])
    sat_p1 = PropsSI('P','T',temperature_K_data[k],'Q',1,CAS_from_any(components[0]))
    y_calc = x_data[k] * gamma_1 * sat_p1 / pressure_Pa
    y_calc_MeOH.append(y_calc)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(x_data, y_data, color='red', label='Experimental data', zorder=3)
plt.plot(x_data, y_calc_MeOH, color='blue', label='Wilson model (calculated)', linewidth=2)
plt.plot(x_data, x_data, color='green', linestyle='--', label='y=x line')

plt.xlabel(f"x_{components[0]}")
plt.ylabel(f"y_{components[0]}")
plt.title("VLE Isobaric Data: {comp1} - {comp2} at P = {P:3.1f} atm".format(comp1=components[0], comp2=components[1], P=pressure_Pa/1e5))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



pass