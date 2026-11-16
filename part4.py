
import csv
import os
import math
import numpy as np


# ==================================================
# LOAD DATA
# ==================================================

with open("data/records.csv", "r", newline="") as file:
    records = list(csv.DictReader(file))

feature_columns = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score",
    "reopened_flag",
    "self_service_first_flag"
]

X = np.array(
    [[float(row[c]) for c in feature_columns] for row in records],
    dtype=float
)

y = np.array([int(row["escalated"]) for row in records])

os.makedirs("outputs", exist_ok=True)

results = []


def save_results():
    with open("outputs/part4_results.txt", "w") as file:
        for key, val in results:
            if isinstance(val, str):
                file.write(f"{key} = {val}\n")
            elif isinstance(val, (int, np.integer)):
                file.write(f"{key} = {int(val)}\n")
            else:
                file.write(f"{key} = {float(val):.4f}\n")


def write_csv(filename, header, rows):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


# ==================================================
# RECREATE THE PART 2 TRAIN / TEST SPLIT
# ==================================================

train_indices = []
test_indices = []

for class_value in [0, 1]:
    indices = np.where(y == class_value)[0]
    split_position = (7 * len(indices)) // 10

    train_indices.extend(indices[:split_position])
    test_indices.extend(indices[split_position:])

train_indices = sorted(train_indices)
test_indices = sorted(test_indices)

X_train = X[train_indices]
X_test = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

# Standardise using training data only
train_mean = np.mean(X_train, axis=0)
train_sd = np.std(X_train, axis=0, ddof=1)

X_train_s = (X_train - train_mean) / train_sd
X_test_s = (X_test - train_mean) / train_sd


# ==================================================
# RECREATE PART 2 LOGISTIC REGRESSION
# ==================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_loss(y_true, probabilities):
    probabilities = np.clip(probabilities, 1e-12, 1 - 1e-12)

    return -np.mean(
        y_true * np.log(probabilities)
        + (1 - y_true) * np.log(1 - probabilities)
    )


weights = np.zeros(7)
bias = 0.0
learning_rate = 0.1

for epoch in range(1000):

    probabilities = sigmoid(X_train_s @ weights + bias)
    error = probabilities - y_train

    grad_w = (X_train_s.T @ error) / len(y_train)
    grad_b = np.mean(error)

    weights -= learning_rate * grad_w
    bias -= learning_rate * grad_b

logistic_train_prob = sigmoid(X_train_s @ weights + bias)
logistic_test_prob = sigmoid(X_test_s @ weights + bias)


# ==================================================
# RECREATE PART 2 KNN
# ==================================================

def confusion_metrics(actual, predicted):
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return tp, fp, tn, fn, precision, recall, f1


def knn_probabilities(reference_X, reference_y, query_X, k):
    probabilities = []

    for query in query_X:
        distances = np.sum((reference_X - query) ** 2, axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]

        probabilities.append(np.mean(reference_y[nearest]))

    return np.array(probabilities)


# Find best K using the five folds
best_k = None
best_f1 = -1

for k in range(1, 22, 2):

    fold_scores = []

    for fold in range(5):

        validation_positions = np.arange(len(y_train)) % 5 == fold
        reference_positions = ~validation_positions

        fold_prob = knn_probabilities(
            X_train_s[reference_positions],
            y_train[reference_positions],
            X_train_s[validation_positions],
            k
        )

        fold_predictions = (fold_prob >= 0.5).astype(int)

        metrics = confusion_metrics(
            y_train[validation_positions],
            fold_predictions
        )

        fold_scores.append(metrics[6])

    mean_f1 = np.mean(fold_scores)

    if mean_f1 > best_f1:
        best_f1 = mean_f1
        best_k = k

knn_test_prob = knn_probabilities(
    X_train_s,
    y_train,
    X_test_s,
    best_k
)


# ==================================================
# TASK 4.1 - ENTROPY AND GINI SPLITS
# ==================================================

def entropy(labels):
    if len(labels) == 0:
        return 0.0

    p = np.mean(labels)

    if p == 0 or p == 1:
        return 0.0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def gini(labels):
    if len(labels) == 0:
        return 0.0

    p = np.mean(labels)
    return 1 - p ** 2 - (1 - p) ** 2


root_entropy = entropy(y_train)
root_gini = gini(y_train)

results.append(("root_entropy", root_entropy))
results.append(("root_gini", root_gini))

# Original, unstandardised training features
split_conditions = {
    "S1": X_train[:, 5] == 1,
    "S2": X_train[:, 6] == 1,
    "S3": X_train[:, 2] > 0.5,
    "S4": X_train[:, 0] <= 24
}

split_rows = []
split_gains = {}

for split_name, yes_mask in split_conditions.items():

    no_mask = ~yes_mask

    yes_y = y_train[yes_mask]
    no_y = y_train[no_mask]

    n_yes = len(yes_y)
    n_no = len(no_y)

    positive_yes = int(np.sum(yes_y))
    positive_no = int(np.sum(no_y))

    weighted_entropy = (
        n_yes * entropy(yes_y) + n_no * entropy(no_y)
    ) / len(y_train)

    weighted_gini = (
        n_yes * gini(yes_y) + n_no * gini(no_y)
    ) / len(y_train)

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_gains[split_name] = (info_gain, gini_gain)

    split_rows.append([
        split_name,
        n_yes,
        positive_yes,
        n_no,
        positive_no,
        f"{info_gain:.4f}",
        f"{gini_gain:.4f}"
    ])

write_csv(
    "outputs/part4_splits.csv",
    [
        "split", "n_yes", "positives_yes",
        "n_no", "positives_no",
        "info_gain", "gini_gain"
    ],
    split_rows
)

best_info_split = max(
    split_gains,
    key=lambda name: split_gains[name][0]
)

best_gini_split = max(
    split_gains,
    key=lambda name: split_gains[name][1]
)

results.append(("best_split_by_info_gain", best_info_split))
results.append(("best_split_by_gini_gain", best_gini_split))


# ==================================================
# TASK 4.2 - DECISION STUMP AND VOTING
# ==================================================

stump_mask_train = split_conditions[best_info_split]

stump_rate_yes = np.mean(y_train[stump_mask_train])
stump_rate_no = np.mean(y_train[~stump_mask_train])

results.append(("stump_rate_yes", stump_rate_yes))
results.append(("stump_rate_no", stump_rate_no))

# Apply the same split condition to test data
if best_info_split == "S1":
    stump_mask_test = X_test[:, 5] == 1
elif best_info_split == "S2":
    stump_mask_test = X_test[:, 6] == 1
elif best_info_split == "S3":
    stump_mask_test = X_test[:, 2] > 0.5
else:
    stump_mask_test = X_test[:, 0] <= 24

stump_test_prob = np.where(
    stump_mask_test,
    stump_rate_yes,
    stump_rate_no
)

# Three individual models
logistic_pred = (logistic_test_prob >= 0.3).astype(int)

knn_pred = (
    knn_test_prob >= (best_k + 1) / (2 * best_k)
).astype(int)

stump_pred = (stump_test_prob >= 0.3).astype(int)

# Hard voting
hard_vote_pred = (
    logistic_pred + knn_pred + stump_pred >= 2
).astype(int)

# Soft voting
soft_vote_prob = (
    logistic_test_prob + knn_test_prob + stump_test_prob
) / 3

soft_vote_pred = (soft_vote_prob >= 0.3).astype(int)

voting_models = [
    ("logistic", logistic_pred),
    ("knn", knn_pred),
    ("stump", stump_pred),
    ("hard_vote", hard_vote_pred),
    ("soft_vote", soft_vote_pred)
]

voting_rows = []

for model_name, predictions in voting_models:

    tp, fp, tn, fn, precision, recall, f1 = confusion_metrics(
        y_test, predictions
    )

    voting_rows.append([
        model_name,
        tp, fp, tn, fn,
        f"{precision:.4f}",
        f"{recall:.4f}",
        f"{f1:.4f}"
    ])

write_csv(
    "outputs/part4_voting.csv",
    [
        "model", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1"
    ],
    voting_rows
)


# ==================================================
# TASK 4.3 - TWO ROUNDS OF ADABOOST
# ==================================================

true_y = np.array([1, 1, 1, -1, -1, -1, 1, -1])

h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])

sample_weights = np.full(8, 1 / 8)

adaboost_rows = []

alphas = []

for round_no, prediction in [(1, h1), (2, h2)]:

    weights_before = sample_weights.copy()

    error = np.sum(
        weights_before[prediction != true_y]
    )

    alpha = 0.5 * math.log((1 - error) / error)
    alphas.append(alpha)

    unnormalised = (
        weights_before
        * np.exp(-alpha * true_y * prediction)
    )

    sample_weights = unnormalised / np.sum(unnormalised)

    for i in range(8):
        adaboost_rows.append([
            round_no,
            i + 1,
            f"{weights_before[i]:.4f}",
            f"{unnormalised[i]:.4f}",
            f"{sample_weights[i]:.4f}"
        ])

    results.append((f"round{round_no}_error", error))
    results.append((f"round{round_no}_alpha", alpha))

ensemble_scores = alphas[0] * h1 + alphas[1] * h2

ensemble_predictions = np.where(ensemble_scores >= 0, 1, -1)

ensemble_accuracy = np.mean(ensemble_predictions == true_y)

results.append(("ensemble_accuracy", ensemble_accuracy))

write_csv(
    "outputs/part4_adaboost.csv",
    [
        "round", "point", "weight_before",
        "weight_unnormalised", "weight_after"
    ],
    adaboost_rows
)


# ==================================================
# TASK 4.4 - OUT OF BAG FRACTION
# ==================================================

oob_fraction = (1 - 1 / 560) ** 560

results.append(("oob_fraction", oob_fraction))


# ==================================================
# TASK 4.5 - NEURAL NETWORK PARAMETERS
# ==================================================

# Model A: Fully connected network
mlp_hidden1 = (784 + 1) * 16
mlp_hidden2 = (16 + 1) * 16
mlp_output = (16 + 1) * 10

mlp_total = mlp_hidden1 + mlp_hidden2 + mlp_output

# Model A: Convolutional network
cnn_conv1 = (5 * 5 * 1 + 1) * 6
cnn_conv2 = (5 * 5 * 6 + 1) * 16

# Spatial sizes:
# 28x28x1 -> conv1 -> 28x28x6
# 28x28x6 -> pool -> 14x14x6
# 14x14x6 -> conv2 -> 10x10x16
# 10x10x16 -> pool -> 5x5x16
# Flatten -> 400

cnn_dense1 = (400 + 1) * 32
cnn_dense2 = (32 + 1) * 10

cnn_total = (
    cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2
)

architecture_rows = [
    ["mlp_hidden1", mlp_hidden1],
    ["mlp_hidden2", mlp_hidden2],
    ["mlp_output", mlp_output],
    ["cnn_conv1", cnn_conv1],
    ["cnn_conv2", cnn_conv2],
    ["cnn_dense1", cnn_dense1],
    ["cnn_dense2", cnn_dense2]
]

write_csv(
    "outputs/part4_architecture.csv",
    ["layer", "parameters"],
    architecture_rows
)

results.append(("mlp_total", mlp_total))
results.append(("cnn_total", cnn_total))

# Fully connected versus convolutional layer
fc_layer_params = (128 * 128 * 3 + 1) * 1024
conv_layer_params = (3 * 3 * 3 + 1) * 64

results.append(("fc_layer_params", fc_layer_params))
results.append(("conv_layer_params", conv_layer_params))

# Model B: RNN
wxh = 64 * 128
whh = 128 * 128
why = 128 * 16
bh = 128
by = 16

rnn_total = wxh + whh + why + bh + by

results.append(("rnn_total", rnn_total))


# ==================================================
# SAVE RESULTS
# ==================================================

save_results()

print("Part 4 completed successfully!")
print("Results saved in outputs/part4_results.txt")
print("Splits saved in outputs/part4_splits.csv")
print("Voting saved in outputs/part4_voting.csv")
print("AdaBoost saved in outputs/part4_adaboost.csv")
print("Architecture saved in outputs/part4_architecture.csv")
