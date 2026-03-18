import numpy as np
import matplotlib.pyplot as plt

# 1. Setup True Parameters and Simulation Settings
true_mu = 1.0
true_b = 2.0
M = 500  # Number of experiments per N
N_values = [10, 50, 100, 500, 1000, 5000]

mu_estimates_all = []
b_estimates_all = []
np.random.seed(42)

# Generate Data
for N in N_values:
    samples = np.random.laplace(loc=true_mu, scale=true_b, size=(M, N))
    mu_hat = np.median(samples, axis=1)
    b_hat = np.mean(np.abs(samples - mu_hat[:, np.newaxis]), axis=1)
    mu_estimates_all.append(mu_hat)
    b_estimates_all.append(b_hat)

# 2. Setup Figure and Styling
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Define styling dictionaries for Plot 1 (Location, blue theme)
box_style_mu = dict(facecolor="#A2D2FF", color="#023E8A", linewidth=1.5)
median_style = dict(color="#D62828", linewidth=2.5)  # Shared red median line
whisker_style_mu = dict(color="#023E8A", linewidth=1.5)
cap_style_mu = dict(color="#023E8A", linewidth=1.5)
flier_style_mu = dict(
    marker="o",
    markerfacecolor="#023E8A",
    alpha=0.3,
    markersize=4,
    markeredgecolor="none",
)

# --- Plot 1: Location Parameter (mu) ---
bplot1 = ax1.boxplot(
    mu_estimates_all,
    tick_labels=N_values,
    patch_artist=True,
    boxprops=box_style_mu,
    medianprops=median_style,
    whiskerprops=whisker_style_mu,
    capprops=cap_style_mu,
    flierprops=flier_style_mu,
)

# Add the "True Value" line and styling
ax1.axhline(
    y=true_mu,
    color="#D62828",
    linestyle="--",
    linewidth=2,
    zorder=0,
    label=r"True $\mu$ = " + f"{true_mu}",
)
ax1.set_title(
    r"MLE of Location Parameter ($\mu$) over " + f"{M} trials",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
ax1.set_xlabel("Sample Size (N)", fontsize=12, fontweight="bold")
ax1.set_ylabel(r"Estimate Value ($\hat{\mu}$)", fontsize=12, fontweight="bold")
ax1.legend(loc="upper right", fontsize=11, frameon=True, shadow=True, borderpad=1)
ax1.grid(axis="y", linestyle="--", alpha=0.7)
ax1.set_facecolor("#F8F9FA")

# --- Plot 2: Scale Parameter (b) ---
# Define styling dictionaries for Plot 2 (Scale, green theme)
box_style_b = dict(facecolor="#B7E4C7", color="#1B4332", linewidth=1.5)
whisker_style_b = dict(color="#1B4332", linewidth=1.5)
cap_style_b = dict(color="#1B4332", linewidth=1.5)
flier_style_b = dict(
    marker="o",
    markerfacecolor="#1B4332",
    alpha=0.3,
    markersize=4,
    markeredgecolor="none",
)

bplot2 = ax2.boxplot(
    b_estimates_all,
    tick_labels=N_values,
    patch_artist=True,
    boxprops=box_style_b,
    medianprops=median_style,
    whiskerprops=whisker_style_b,
    capprops=cap_style_b,
    flierprops=flier_style_b,
)

# Add the "True Value" line and styling
ax2.axhline(
    y=true_b,
    color="#D62828",
    linestyle="--",
    linewidth=2,
    zorder=0,
    label=r"True $b$ = " + f"{true_b}",
)
ax2.set_title(
    r"MLE of Scale Parameter ($b$) over " + f"{M} trials",
    fontsize=14,
    pad=15,
    fontweight="bold",
)
ax2.set_xlabel("Sample Size (N)", fontsize=12, fontweight="bold")
ax2.set_ylabel(r"Estimate Value ($\hat{b}$)", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", fontsize=11, frameon=True, shadow=True, borderpad=1)
ax2.grid(axis="y", linestyle="--", alpha=0.7)
ax2.set_facecolor("#F8F9FA")

# Final layout adjustments
plt.tight_layout()
plt.show()
