import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the system of first-order ODEs
def system(eta, y, params):
    f, f_prime, f_double_prime, F, F_prime, theta, theta_prime, theta_p, theta_p_prime = y
    l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T, S = params
    
    # Define the derivatives based on the provided equations
    d_f = f_prime
    d_f_prime = f_double_prime
    d_f_double_prime = -f * f_double_prime + (f_prime)**2 - l * (F_prime - f_prime) + S * f_prime - Q * np.exp(-A * eta)
    
    d_F = F_prime
    d_F_prime = (F**2 - beta * (F_prime - f_prime)) / F if F != 0 else 0  # Avoid division by zero
    
    d_theta = theta_prime
    d_theta_prime = (-Pr * (f * theta_prime - 2 * f_prime * theta) 
                     - Pr * l * beta_T * (theta_p - theta) 
                     - l * beta * Pr * Ec * (F_prime - f_prime)**2) / (1 + 4 * Rd / 3)
    
    d_theta_p = theta_p_prime
    d_theta_p_prime = (2 * F_prime * theta_p - beta_T * (theta_p - theta)) / F if F != 0 else 0  # Avoid division by zero
    
    return [d_f, d_f_prime, d_f_double_prime, d_F, d_F_prime, d_theta, d_theta_prime, d_theta_p, d_theta_p_prime]

# Define initial conditions for the boundary conditions
def boundary_conditions(params):
    l, beta, Q, A, Rd, Pr, Bi, Ec, beta_T = params

    f0 = 0            # f(0) = 0
    f_prime0 = 1      # f'(0) = 1
    theta0 = 1        
    theta_prime0 = -Bi * (1 - theta0)  # θ'(0) = -Bi * (1 - θ(0))
    
    f_double_prime0 = 0
    F0 = 1e-6  
    F_prime0 = 0
    theta_p0 = 0
    theta_p_prime0 = 0

    return [f0, f_prime0, f_double_prime0, F0, F_prime0, theta0, theta_prime0, theta_p0, theta_p_prime0]

params = (0.6, 0.5, 0.1, 2.0, 3.0, 0.1, 1.5, 0.5, 0.5, 1.0)
S = 2.0  # Assuming S = 2 as a constant (can be adjusted if needed)

# Integration range for eta
eta_span = (0, 5)

# Set `l` to the last element of `params`
l = params[-1]

# Prepare the parameters for the `system` function by adding `S` at the end
full_params = (*params[:-1], S)

# Initial conditions
y0 = boundary_conditions(params[:-1])  # Exclude `l` as the last parameter in boundary conditions
solution = solve_ivp(system, eta_span, y0, args=(full_params,), method='RK45', dense_output=True, rtol=1e-6, atol=1e-9)

# Extract eta (independent variable) and the derivatives
eta_vals = solution.t
f_prime_vals = solution.y[1]       # f'(eta)
F_prime_vals = solution.y[4]       # F'(eta)
theta_vals = solution.y[5]          # θ(eta) (used as Q)
theta_p_vals = solution.y[7]        # θ_p(eta) (used as Q_p)
f_vals = -solution.y[0]       # -f(eta)
F_vals = -solution.y[3]       # -F(eta)


# Plotting all on a single graph
plt.figure(figsize=(14, 8))

# Plot with negative eta values for mirroring
plt.plot(-eta_vals, f_prime_vals/100000, label="f'(η)", linestyle='-', color='blue')
plt.plot(-eta_vals, F_prime_vals/100000, label="F'(η)", linestyle='-', color='orange')
plt.plot(-eta_vals, -theta_vals/100000, label="Q'(η)", linestyle='-', color='green')    # Assuming Q is θ
plt.plot(-eta_vals, theta_p_vals/100000, label="Q_p'(η)", linestyle='-', color='red')  # Assuming Q_p is θ_p
plt.plot(-eta_vals, -f_vals/100000, label='f(η)', linestyle='-', color='black')         # Remove negative sign in label
plt.plot(-eta_vals, -F_vals/100000, label='F(η)', linestyle='-', color='yellow')       # Remove negative sign in label
plt.plot(-eta_vals, -theta_vals/100000, label='θ(η)', linestyle='-', color='pink')     # Remove negative sign in label
plt.plot(-eta_vals, theta_p_vals/100000, label='θ_p(η)', linestyle='-', color='purple') # Remove negative sign in label

# Labels and title
plt.xlabel('η')
plt.ylabel('Values')
plt.title("Plot of f'(η), F'(η), Q'(η), and Q_p'(η)")
plt.grid(True)
plt.legend()

plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=True)

plt.tight_layout()
plt.show()

# Calculate f''(0) and q'(0)
f_double_prime_0 = y0[2]  # f''(0)
theta_prime_0 = y0[6]      # q'(0) (theta_prime0)

# Print the values
print(f"f''(0) = {f_double_prime_0}")
print(f"q'(0) = {theta_prime_0}")

