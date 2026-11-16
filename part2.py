
import csv
import os
import math
import numpy as np

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

feature_names = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score",
    "reopened_flag",
    "self_service_first_flag"
]

with open("data/records.csv", "r", newline="") as f:
    rows = list(csv.DictReader(f))

X = np.array([
    [float(row[name]) for name in feature_names]
    for row in rows
])

y = np.array([int(row["escalated"]) for row in rows])

# --------------------------------------------------
# 2. STRATIFIED 70/30 SPLIT
# --------------------------------------------------

zero_indices = [i for i in range(len(y)) if y[i] == 0]
one_indices = [i for i in range(len(y)) if y[i] == 1]

zero_cut = 7 * len(zero_indices) // 10
one_cut = 7 * len(one_indices) // 10

train_indices = sorted(
    zero_indices[:zero_cut] + one_indices[:one_cut]
)
test_indices = sorted(
    zero_indices[zero_cut:] + one_indices[one_cut:]
)

X_train_raw = X[train_indices]
X_test_raw = X[test_indices]
y_train = y[train_indices]
y_test = y[test_indices]

# Standardise using training data only
train_mean = np.mean(X_train_raw, axis=0)
train_sd = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - train_mean) / train_sd
X_test = (X_test_raw - train_mean) / train_sd

results = {}

results["train_rows"] = len(y_train)
results["train_positives"] = int(np.sum(y_train))
results["test_rows"] = len(y_test)
results["test_positives"] = int(np.sum(y_test))

# --------------------------------------------------
# 3. LOGISTIC REGRESSION FUNCTIONS
# --------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def binary_loss(actual, probability):
    probability = np.clip(probability, 1e-12, 1 - 1e-12)
    return -np.mean(
        actual * np.log(probability)
        + (1 - actual) * np.log(1 - probability)
    )


def train_logistic(features, target, learning_rate, updates,
                   checkpoints=None):

    weights = np.zeros(features.shape[1])
    bias = 0.0
    losses = {}

    if checkpoints is None:
        checkpoints = []

    if 0 in checkpoints:
        losses[0] = binary_loss(
            target, sigmoid(features @ weights + bias)
        )

    for epoch in range(1, updates + 1):
        probability = sigmoid(features @ weights + bias)
        error = probability - target

        grad_weights = features.T @ error / len(target)
        grad_bias = np.mean(error)

        weights -= learning_rate * grad_weights
        bias -= learning_rate * grad_bias

        if epoch in checkpoints:
            new_probability = sigmoid(features @ weights + bias)
            losses[epoch] = binary_loss(target, new_probability)

    return weights, bias, losses


# Main model: 1000 updates
main_epochs = [0, 1, 10, 100, 1000]

weights, bias, main_losses = train_logistic(
    X_train, y_train, 0.1, 1000, main_epochs
)

for epoch in main_epochs:
    results[f"loss_epoch_{epoch}"] = main_losses[epoch]

for i, weight in enumerate(weights, start=1):
    results[f"weight_X{i}"] = weight
    results[f"odds_ratio_X{i}"] = math.exp(float(weight))

results["bias"] = bias

test_probability = sigmoid(X_test @ weights + bias)
results["test_bce"] = binary_loss(y_test, test_probability)

# --------------------------------------------------
# 4. LEARNING RATE COMPARISON
# --------------------------------------------------

for rate in [0.01, 0.1, 1.0]:
    _, _, rate_losses = train_logistic(
        X_train, y_train, rate, 200, [200]
    )
    results[f"loss_epoch_200_eta_{rate}"] = rate_losses[200]

# --------------------------------------------------
# 5. CONFUSION MATRIX AND METRICS
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    accuracy = (tp + tn) / len(actual)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity
    }


for threshold in [0.5, 0.3]:
    predicted = (test_probability >= threshold).astype(int)
    metrics = calculate_metrics(y_test, predicted)

    for key, value in metrics.items():
        results[f"tau_{threshold}_{key}"] = value

blind_prediction = np.zeros(len(y_test), dtype=int)
results["blind_accuracy"] = np.mean(blind_prediction == y_test)

# --------------------------------------------------
# 6. THRESHOLD SWEEP CSV
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_thresholds.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])

    for threshold in np.arange(0.1, 1.0, 0.1):
        predicted = (test_probability >= threshold).astype(int)
        m = calculate_metrics(y_test, predicted)

        fpr = m["fp"] / (m["fp"] + m["tn"]) if m["fp"] + m["tn"] else 0

        writer.writerow([
            f"{threshold:.4f}",
            m["tp"], m["fp"], m["tn"], m["fn"],
            f"{m['recall']:.4f}",
            f"{fpr:.4f}"
        ])

# --------------------------------------------------
# 7. K-NEAREST NEIGHBOURS WITH 5-FOLD CV
# --------------------------------------------------

def knn_predict(reference_X, reference_y, query_X, k):
    predictions = []
    probabilities = []

    for query in query_X:
        distances = np.sqrt(np.sum((reference_X - query) ** 2, axis=1))
        order = np.argsort(distances, kind="stable")
        neighbour_labels = reference_y[order[:k]]

        positive_count = int(np.sum(neighbour_labels))
        probabilities.append(positive_count / k)
        predictions.append(
            1 if positive_count >= (k + 1) / 2 else 0
        )

    return np.array(predictions), np.array(probabilities)


def simple_f1(actual, predicted):
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    return (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )


knn_scores = []

for k in range(1, 22, 2):
    fold_scores = []

    for fold in range(5):
        validation_positions = [
            i for i in range(len(y_train)) if i % 5 == fold
        ]
        reference_positions = [
            i for i in range(len(y_train)) if i % 5 != fold
        ]

        validation_X = X_train[validation_positions]
        validation_y = y_train[validation_positions]

        reference_X = X_train[reference_positions]
        reference_y = y_train[reference_positions]

        predicted, _ = knn_predict(
            reference_X, reference_y, validation_X, k
        )

        fold_scores.append(simple_f1(validation_y, predicted))

    mean_score = float(np.mean(fold_scores))
    knn_scores.append((k, mean_score))

best_k, best_score = max(
    knn_scores, key=lambda item: (item[1], -item[0])
)

results["knn_best_k"] = best_k

with open("outputs/part2_knn_cv.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "mean_cv_f1"])

    for k, score in knn_scores:
        writer.writerow([k, f"{score:.4f}"])

# Evaluate best K on test data
knn_prediction, knn_probability = knn_predict(
    X_train, y_train, X_test, best_k
)

knn_metrics = calculate_metrics(y_test, knn_prediction)

for key, value in knn_metrics.items():
    if key != "specificity":
        results[f"knn_{key}"] = value

# --------------------------------------------------
# 8. CHOOSE THRESHOLD USING TRAINING COST
# --------------------------------------------------

train_probability = sigmoid(X_train @ weights + bias)

best_cost = float("inf")
best_threshold = None

for k in range(1, 20):
    threshold = k / 20
    predicted = (train_probability >= threshold).astype(int)
    m = calculate_metrics(y_train, predicted)

    cost = 50 * m["fp"] + 400 * m["fn"]

    if cost < best_cost:
        best_cost = cost
        best_threshold = threshold

results["cost_tau"] = best_threshold
results["cost_train_at_tau"] = best_cost

test_at_tau = (test_probability >= best_threshold).astype(int)
test_metrics_at_tau = calculate_metrics(y_test, test_at_tau)

results["cost_test_at_tau"] = (
    50 * test_metrics_at_tau["fp"]
    + 400 * test_metrics_at_tau["fn"]
)

test_at_half = (test_probability >= 0.5).astype(int)
half_metrics = calculate_metrics(y_test, test_at_half)

results["cost_test_at_0.5"] = (
    50 * half_metrics["fp"] + 400 * half_metrics["fn"]
)

results["cost_test_flag_nobody"] = 400 * int(np.sum(y_test))

# --------------------------------------------------
# 9. SAVE RESULTS
# --------------------------------------------------

with open("outputs/part2_results.txt", "w") as f:
    for key, value in results.items():
        if isinstance(value, (int, np.integer)):
            f.write(f"{key} = {int(value)}\n")
        else:
            f.write(f"{key} = {float(value):.4f}\n")

print("Part 2 completed successfully!")
print("Results saved in outputs/part2_results.txt")
print("Threshold CSV saved in outputs/part2_thresholds.csv")
print("KNN CV CSV saved in outputs/part2_knn_cv.csv")
