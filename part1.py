
import csv
import os
import math
import numpy as np

# Load the generated dataset
with open("data/records.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Extract the required columns
X1 = np.array([float(r["customer_tenure_months"]) for r in rows])
X2 = np.array([float(r["earlier_ticket_count_90d"]) for r in rows])
X3 = np.array([float(r["sla_consumed_ratio"]) for r in rows])
X4 = np.array([float(r["account_value_inr_thousands"]) for r in rows])
X5 = np.array([float(r["satisfaction_score"]) for r in rows])
X6 = np.array([float(r["reopened_flag"]) for r in rows])
X7 = np.array([float(r["self_service_first_flag"]) for r in rows])
y = np.array([float(r["resolution_minutes"]) for r in rows])

n = len(y)
results = {}

# PART 1.1: Mean, standard deviation and confidence interval
mean_y = np.mean(y)
sd_y = np.std(y, ddof=1)
se_y = sd_y / math.sqrt(n)

results["n"] = n
results["mean_value"] = mean_y
results["sd_value"] = sd_y
results["se_value"] = se_y
results["ci95_low"] = mean_y - 1.96 * se_y
results["ci95_high"] = mean_y + 1.96 * se_y

# PART 1.2: Hypothesis test
z_stat = (mean_y - 8.0) / se_y
results["z_stat"] = z_stat
results["reject_h0"] = "yes" if abs(z_stat) > 1.96 else "no"

# PART 1.3: Required sample size
n_required = math.ceil((1.96 * sd_y / 0.5) ** 2)
results["n_required"] = n_required

# PART 1.4: Simple regression using X4
mean_x = np.mean(X4)
sd_x = np.std(X4, ddof=1)

reg_slope = np.sum((X4 - mean_x) * (y - mean_y)) / np.sum((X4 - mean_x) ** 2)
reg_intercept = mean_y - reg_slope * mean_x

fitted = reg_intercept + reg_slope * X4
residuals = y - fitted

tss = np.sum((y - mean_y) ** 2)
ess = np.sum((fitted - mean_y) ** 2)
rss = np.sum(residuals ** 2)

results["reg_slope"] = reg_slope
results["reg_intercept"] = reg_intercept
results["tss"] = tss
results["ess"] = ess
results["rss"] = rss
results["tss_equals_ess_plus_rss"] = (
    "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"
)
results["r2"] = 1 - rss / tss
results["rmse"] = math.sqrt(rss / n)
results["mae"] = np.mean(np.abs(residuals))

# PART 1.5: Gradient descent
z = (X4 - mean_x) / sd_x
m = 0.0
c = 0.0
eta = 0.1

def gd_cost(m, c):
    error = y - (m * z + c)
    return np.mean(error ** 2)

results["gd_J0"] = gd_cost(m, c)

for epoch in range(1, 501):
    prediction = m * z + c
    error = y - prediction

    dm = (-2 / n) * np.sum(z * error)
    dc = (-2 / n) * np.sum(error)

    m = m - eta * dm
    c = c - eta * dc

    if epoch in [1, 10, 100, 500]:
        results["gd_J" + str(epoch)] = gd_cost(m, c)

gd_slope = m / sd_x
gd_intercept = c - m * mean_x / sd_x

results["gd_slope"] = gd_slope
results["gd_intercept"] = gd_intercept
results["gd_slope_gap"] = abs(gd_slope - reg_slope)

# PART 1.6: Residual spread and log-log regression
order = np.argsort(fitted, kind="stable")
lower = order[:400]
upper = order[400:]

sd_lower = np.std(residuals[lower], ddof=1)
sd_upper = np.std(residuals[upper], ddof=1)

results["resid_sd_lower_raw"] = sd_lower
results["resid_sd_upper_raw"] = sd_upper
results["resid_sd_ratio_raw"] = sd_upper / sd_lower

# Log-log regression
log_x = np.log(X4)
log_y = np.log(y)

mean_lx = np.mean(log_x)
mean_ly = np.mean(log_y)

log_slope = np.sum((log_x - mean_lx) * (log_y - mean_ly)) / np.sum((log_x - mean_lx) ** 2)
log_intercept = mean_ly - log_slope * mean_lx

log_fitted = log_intercept + log_slope * log_x
log_residuals = log_y - log_fitted

log_tss = np.sum((log_y - mean_ly) ** 2)
log_rss = np.sum(log_residuals ** 2)
log_r2 = 1 - log_rss / log_tss
log_rmse = math.sqrt(log_rss / n)

results["loglog_intercept"] = log_intercept
results["loglog_slope"] = log_slope
results["loglog_r2"] = log_r2
results["loglog_rmse"] = log_rmse

log_order = np.argsort(log_fitted, kind="stable")
log_lower = log_order[:400]
log_upper = log_order[400:]

log_sd_lower = np.std(log_residuals[log_lower], ddof=1)
log_sd_upper = np.std(log_residuals[log_upper], ddof=1)

results["resid_sd_lower_log"] = log_sd_lower
results["resid_sd_upper_log"] = log_sd_upper
results["resid_sd_ratio_log"] = log_sd_upper / log_sd_lower

# PART 1.7: Multiple regression using the normal equation
log_X = np.column_stack((
    np.ones(n),
    np.log(X4),
    X1, X2, X3, X5, X6, X7
))

beta = np.linalg.solve(log_X.T @ log_X, log_X.T @ log_y)

mr_fitted = log_X @ beta
mr_residuals = log_y - mr_fitted

mr_tss = np.sum((log_y - mean_ly) ** 2)
mr_rss = np.sum(mr_residuals ** 2)
mr_r2 = 1 - mr_rss / mr_tss

mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1)
loglog_adj_r2 = 1 - (1 - log_r2) * (n - 1) / (n - 1 - 1)

for i in range(8):
    results["mr_beta_" + str(i)] = beta[i]

results["mr_r2"] = mr_r2
results["mr_adj_r2"] = mr_adj_r2
results["loglog_adj_r2"] = loglog_adj_r2

# Save all results
os.makedirs("outputs", exist_ok=True)

with open("outputs/part1_results.txt", "w") as f:
    for key, value in results.items():
        if isinstance(value, str):
            formatted = value
        elif isinstance(value, (int, np.integer)):
            formatted = str(int(value))
        else:
            formatted = f"{value:.4f}"

        f.write(f"{key} = {formatted}\n")

print("Part 1 completed successfully!")
print("Results saved in outputs/part1_results.txt")
