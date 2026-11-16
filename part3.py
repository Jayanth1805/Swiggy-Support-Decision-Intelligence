
import csv
import os
import numpy as np


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("data/records.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    records = list(reader)

columns = [
    "customer_tenure_months",
    "earlier_ticket_count_90d",
    "sla_consumed_ratio",
    "account_value_inr_thousands",
    "satisfaction_score"
]

X = np.array(
    [[float(row[col]) for col in columns] for row in records],
    dtype=float
)

outcome = np.array([int(row["escalated"]) for row in records])
value = np.array([float(row["resolution_minutes"]) for row in records])

n = len(X)

os.makedirs("outputs", exist_ok=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def save_results(results):
    with open("outputs/part3_results.txt", "w") as file:
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


# --------------------------------------------------
# TASK 3.1 - K-MEANS CLUSTERING
# --------------------------------------------------

means = np.mean(X, axis=0)
stds = np.std(X, axis=0, ddof=1)
Z = (X - means) / stds

wcss_values = {}
results = []

for k in range(2, 7):

    step = n // k
    centroids = Z[np.arange(k) * step].copy()

    previous_labels = np.full(n, -1, dtype=int)

    while True:

        distances = np.sum(
            (Z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        labels = np.argmin(distances, axis=1)

        if np.array_equal(labels, previous_labels):
            break

        previous_labels = labels.copy()

        for j in range(k):
            members = Z[labels == j]

            if len(members) > 0:
                centroids[j] = np.mean(members, axis=0)

    final_distances = np.sum(
        (Z - centroids[labels]) ** 2,
        axis=1
    )

    wcss = np.sum(final_distances)
    wcss_values[k] = wcss

    results.append((f"wcss_k{k}", wcss))


# --------------------------------------------------
# TASK 3.2 - ELBOW METHOD
# --------------------------------------------------

elbow_k = 0

for k in range(3, 6):

    previous_drop = wcss_values[k - 1] - wcss_values[k]
    current_drop = wcss_values[k] - wcss_values[k + 1]

    if current_drop < 0.5 * previous_drop:
        elbow_k = k
        break

results.append(("elbow_k", elbow_k))


# --------------------------------------------------
# TASK 3.3 - THREE CUSTOMER SEGMENTS
# --------------------------------------------------

k = 3
step = n // k

centroids = Z[np.arange(k) * step].copy()
previous_labels = np.full(n, -1, dtype=int)

while True:

    distances = np.sum(
        (Z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
        axis=2
    )

    labels = np.argmin(distances, axis=1)

    if np.array_equal(labels, previous_labels):
        break

    previous_labels = labels.copy()

    for j in range(k):
        members = Z[labels == j]

        if len(members) > 0:
            centroids[j] = np.mean(members, axis=0)

# Order clusters by ascending original X3 mean
cluster_order = sorted(
    range(3),
    key=lambda j: np.mean(X[labels == j, 2])
)

segment_names = ["A", "B", "C"]
segment_labels = {}

segment_rows = []

for name, cluster_id in zip(segment_names, cluster_order):

    mask = labels == cluster_id
    segment_labels[name] = mask

    size = int(np.sum(mask))
    feature_means = np.mean(X[mask], axis=0)
    outcome_rate = np.mean(outcome[mask])
    mean_value = np.mean(value[mask])

    results.append((f"size_segment_{name}", size))

    segment_rows.append([
        name,
        size,
        *[f"{v:.4f}" for v in feature_means],
        f"{outcome_rate:.4f}",
        f"{mean_value:.4f}"
    ])

write_csv(
    "outputs/part3_segments.csv",
    [
        "segment", "size",
        "mean_X1", "mean_X2", "mean_X3",
        "mean_X4", "mean_X5",
        "outcome_rate", "mean_value"
    ],
    segment_rows
)


# --------------------------------------------------
# TASK 3.4 - HIERARCHICAL CLUSTERING
# --------------------------------------------------

distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

linkage_rows = []
cut_counts = {}

for linkage in ["single", "complete", "average"]:

    clusters = [[i] for i in range(6)]
    heights = []

    for merge_no in range(1, 6):

        best_distance = float("inf")
        best_pair = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distances = [
                    distance_matrix[a, b]
                    for a in clusters[i]
                    for b in clusters[j]
                ]

                if linkage == "single":
                    distance = min(distances)

                elif linkage == "complete":
                    distance = max(distances)

                else:
                    distance = sum(distances) / len(distances)

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Ensure cluster_a has the smaller first member
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        name_a = "+".join(f"E{x + 1}" for x in sorted(cluster_a))
        name_b = "+".join(f"E{x + 1}" for x in sorted(cluster_b))

        linkage_rows.append([
            linkage,
            merge_no,
            name_a,
            name_b,
            f"{best_distance:.4f}"
        ])

        heights.append(best_distance)

        merged = sorted(cluster_a + cluster_b)

        # Remove the two old clusters and add the merged cluster
        remaining = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        remaining.append(merged)
        clusters = sorted(remaining, key=lambda c: min(c))

    gaps = np.diff(heights)
    largest_gap_index = int(np.argmax(gaps))

    clusters_at_cut = 6 - (largest_gap_index + 1)
    cut_counts[linkage] = clusters_at_cut

write_csv(
    "outputs/part3_hierarchical.csv",
    ["linkage", "merge_no", "cluster_a", "cluster_b", "height"],
    linkage_rows
)

results.append(("single_clusters_at_cut", cut_counts["single"]))
results.append(("complete_clusters_at_cut", cut_counts["complete"]))
results.append(("average_clusters_at_cut", cut_counts["average"]))


# --------------------------------------------------
# TASK 3.5 - PRINCIPAL COMPONENT ANALYSIS (PCA)
# --------------------------------------------------

# Standardised data has a mean vector approximately equal to zero
Z_centered = Z - np.mean(Z, axis=0)

# Covariance matrix
covariance = (Z_centered.T @ Z_centered) / (n - 1)

# Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance)

# Sort eigenvalues from largest to smallest
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Apply sign convention:
# Largest absolute loading in each component must be positive
for j in range(eigenvectors.shape[1]):

    largest_index = np.argmax(np.abs(eigenvectors[:, j]))

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1

eigenvalue_sum = np.sum(eigenvalues)

explained_ratios = eigenvalues / eigenvalue_sum
cumulative_ratios = np.cumsum(explained_ratios)

components_for_80pct = int(
    np.argmax(cumulative_ratios >= 0.8) + 1
)

# Keep the first two principal components
W = eigenvectors[:, :2]

# Project all records
projections = Z_centered @ W

results.append(("eigenvalue_sum", eigenvalue_sum))

for i in range(5):
    results.append((f"eigenvalue_{i + 1}", eigenvalues[i]))

for i in range(5):
    results.append((f"explained_ratio_{i + 1}", explained_ratios[i]))

for i in range(5):
    results.append((f"cumulative_ratio_{i + 1}", cumulative_ratios[i]))

results.append(("components_for_80pct", components_for_80pct))

for i in range(5):
    results.append((f"pc1_loading_X{i + 1}", W[i, 0]))

for i in range(5):
    results.append((f"pc2_loading_X{i + 1}", W[i, 1]))

# First three data-row projections
for i in range(3):
    results.append((f"row{i + 1}_pc1", projections[i, 0]))
    results.append((f"row{i + 1}_pc2", projections[i, 1]))

# Mean PC1 score for each segment
for name in segment_names:
    mean_pc1 = np.mean(projections[segment_labels[name], 0])
    results.append((f"mean_pc1_segment_{name}", mean_pc1))


# --------------------------------------------------
# SAVE ALL RESULTS
# --------------------------------------------------

save_results(results)

print("Part 3 completed successfully!")
print("Results saved in outputs/part3_results.txt")
print("Segments saved in outputs/part3_segments.csv")
print("Hierarchical clustering saved in outputs/part3_hierarchical.csv")
