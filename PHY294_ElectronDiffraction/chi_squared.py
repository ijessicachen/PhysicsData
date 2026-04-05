import numpy as np

x = np.array([2.4530128456020808e-11, 2.2392841156436615e-11, 2.073174243364304e-11,
              1.9392769304383814e-11, 1.8283678241488162e-11, 1.7345420174629407e-11])

y1 = np.array([22.8E-3, 20.5E-3, 19.0E-3, 17.5E-3, 17.0E-3, 16.3E-3])
y2 = np.array([32.8E-3, 29.9E-3, 27.1E-3, 25.3E-3, 23.8E-3, 22.2E-3])

sigma_y = 0.5E-4  # reading uncertainty (same for all points)

sigma_x = np.abs(np.array([-4.90602569e-13, -3.73214019e-13, -2.96167749e-13,
                            -2.42409616e-13, -2.03151980e-13, -1.73454202e-13]))

for label, y in [("y1", y1), ("y2", y2)]:
    m, b = np.polyfit(x, y, 1)
    fit = m * x + b
    residuals = y - fit

    # Propagate x uncertainty into y: sigma_total = sqrt(sigma_y^2 + (m * sigma_x)^2)
    sigma_total = np.sqrt(sigma_y**2 + (m * sigma_x)**2)

    chi2 = np.sum((residuals / sigma_total)**2)
    chi2_red = chi2 / (len(y) - 2)  # degrees of freedom = N - 2 for linear fit

    print(f"{label}: m = {m:.4e}, b = {b:.4e}")
    print(f"  sigma_total: {sigma_total}")
    print(f"  chi2 = {chi2:.4f}, chi2_red = {chi2_red:.4f}\n")
