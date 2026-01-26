import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1. Define the Reference Phase Field Function
def LFP_ocp_phase_field(sto):
    # Prevent log(0) errors by clipping sto strictly between 0 and 1
    sto = np.clip(sto, 1e-6, 1-1e-6)
    return 3.43 - 0.0257 * (np.log(sto/(1-sto)) + 3.8 * (1 - 2*sto))

# 2. Define the General Parametric Model
def parametric_ocp(sto, p):
    # p = [const, slope, pre_exp1, arg_exp1, pre_exp2, arg_exp2]
    return p[0] + p[1]*sto + p[2]*np.exp(p[3]*sto) + p[4]*np.exp(p[5]*(1-sto))

# 3. Optimization Routine (Delithiation Only)
def optimize_delithiation():
    
    # --- Settings ---
    FIXED_SLOPE = -0.01
    
    # --- Constraint Targets ---
    sto_low = 0.001
    sto_mid = 0.84   # New constraint
    sto_mid_up = 0.97 # Slightly above mid to avoid numerical issues
    sto_high = 0.99
    
    target_low = LFP_ocp_phase_field(sto_low)
    target_mid = LFP_ocp_phase_field(sto_mid)
    target_mid_up = LFP_ocp_phase_field(sto_mid_up)
    target_high = LFP_ocp_phase_field(sto_high)

    # Fit Region: 0.85 to 1.0
    fit_sto = np.linspace(0.85, 0.999, 100)
    fit_targets = LFP_ocp_phase_field(fit_sto)

    # Initial Guesses (Perturbed to show convergence)
    # x = [const, pre1, arg1, pre2, arg2]
    x0 = [3.46, 0.1, -1000.0, -0.1, -40.0]

    # --- Wrapper to handle fixed slope ---
    def get_full_p(x):
        return [x[0], FIXED_SLOPE, x[1], x[2], x[3], x[4]]

    # --- Objective Function ---
    def objective(x):
        p = get_full_p(x)
        model_vals = parametric_ocp(fit_sto, p)
        # Weight the error to prioritize shape between the constraints
        return np.sum((model_vals - fit_targets)**2) * 1e4

    # --- Constraints ---
    def constraint_low(x): # sto = 0.005
        return parametric_ocp(sto_low, get_full_p(x)) - target_low
    
    def constraint_mid(x): # sto = 0.85
        return parametric_ocp(sto_mid, get_full_p(x)) - target_mid

    def constraint_high(x): # sto = 0.995
        return parametric_ocp(sto_high, get_full_p(x)) - target_high
    
    def constraint_mid_up(x): # sto = 0.9
        return parametric_ocp(sto_mid_up, get_full_p(x)) - target_mid_up

    cons = (
        {'type': 'eq', 'fun': constraint_low},
        {'type': 'eq', 'fun': constraint_mid}, # Added constraint
        {'type': 'eq', 'fun': constraint_high},
        {'type': 'eq', 'fun': constraint_mid_up}  # Added constraint
    )

    # --- Bounds ---
    bnds = (
        (3.3, 3.6),      # const
        (0.01, 5),     # pre_exp1 (positive)
        (-500, -1),     # arg_exp1 (decay) -> p[3]
        (-5, -0.01),   # pre_exp2 (negative)
        (-500, -1)      # arg_exp2 (decay) -> p[5]
    )

    # Run Optimization
    result = minimize(objective, x0, method='SLSQP', constraints=cons, bounds=bnds, tol=1e-12, options={'maxiter': 1000})
    
    return get_full_p(result.x)

# 4. Run Optimization
print("Optimizing Delithiation with 3 constraints...")
p_delithi = optimize_delithiation()

# 5. Output the Results
print("\n" + "="*40)
print("OPTIMIZED COEFFICIENTS (Delithiation)")
print("="*40)

print(f"c1 = {p_delithi[3]:.4f} * sto")
print(f"c2 = {p_delithi[5]:.4f} * (1 - sto)")
print(f"k = {p_delithi[0]:.4f} + ({p_delithi[1]:.4f} * sto) + ({p_delithi[2]:.4f} * np.exp(c1)) + ({p_delithi[4]:.4f} * np.exp(c2))")

print("\n" + "="*40)
print("FINAL PARAMETRIC FORMULA (Lithiation)")
print(f"c1 = {p_delithi[5]:.4f} * sto")
print(f"c2 = {p_delithi[3]:.4f} * (1 - sto)")
print(f"k = {p_delithi[0]-0.06:.4f} + ({p_delithi[1]:.4f} * sto) + ({-p_delithi[4]:.4f} * np.exp(c1)) + (-{p_delithi[2]:.4f} * np.exp(c2))")


# 6. Visualization
sto_range = np.linspace(0.001, 0.999, 500)
y_phase = LFP_ocp_phase_field(sto_range)
y_delithi_opt = parametric_ocp(sto_range, p_delithi)

plt.figure(figsize=(10, 6))

# Plot Phase Field Reference
plt.plot(sto_range, y_phase, 'k--', label='Phase Field (Target)', linewidth=2)

# Plot Optimized Delithiation
plt.plot(sto_range, y_delithi_opt, 'r-', label='Optimized Delithiation', alpha=0.9, linewidth=2)

# Highlight Fit Region
plt.axvspan(0.85, 1.0, color='red', alpha=0.1, label='Fit Region (0.85-1.0)')

# Plot Constraint Points
pts = [0.005, 0.85, 0.995]
plt.scatter(pts, LFP_ocp_phase_field(np.array(pts)), color='lime', zorder=10, label='Constraints', edgecolors='black', s=80)

plt.xlabel('Stoichiometry (sto)')
plt.ylabel('Potential (V)')
plt.title('Optimized Delithiation (Fixed Slope=-0.01, 3 Constraints)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()