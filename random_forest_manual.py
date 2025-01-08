# prompt: buatkan kode seperti kode "punyaku" namun tanpa menggunakan penanganan data imbalance

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from collections import Counter
import pickle

# Load the cleaned dataset
data_path = 'cleaned_data.csv'
df_cleaned = pd.read_csv(data_path)

# Display the first few rows of the dataset
# df_cleaned.head()

# df_cleaned.columns

# print(df_cleaned.dtypes)

# Split the data into features (X) and target (y)
X = df_cleaned.drop('Target', axis=1)  # All columns except 'Target'
y = df_cleaned['Target']               # The 'Target' column

df_cleaned['Target'].value_counts().plot(kind='bar', figsize=(10,6))
# plt.xticks(rotation=0)
# plt.show()


"""normalisasi"""

# Standardize numerical columns
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


# Function to calculate Gini Impurity
def gini_impurity(y):
    counts = Counter(y)
    total = len(y)
    return 1.0 - sum((count / total) ** 2 for count in counts.values())

# Function to split data
def split_dataset(X, y, feature_index, threshold):
    left_indices = X[:, feature_index] <= threshold
    right_indices = X[:, feature_index] > threshold
    return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

# Class for Decision Tree Node
class DecisionNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    # Function to build decision tree
def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    if depth == max_depth or len(set(y)) == 1 or len(y) < min_samples_split:
        return DecisionNode(value=Counter(y).most_common(1)[0][0])

    n_samples, n_features = X.shape
    best_gini = float('inf')
    best_split = None

    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            X_left, X_right, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                continue

            gini = (len(y_left) / n_samples) * gini_impurity(y_left) + \
                   (len(y_right) / n_samples) * gini_impurity(y_right)

            if gini < best_gini:
                best_gini = gini
                best_split = {
                    'feature': feature_index,
                    'threshold': threshold,
                    'X_left': X_left,
                    'X_right': X_right,
                    'y_left': y_left,
                    'y_right': y_right
                }

    if best_split is None:
        return DecisionNode(value=Counter(y).most_common(1)[0][0])

    left_subtree = build_tree(best_split['X_left'], best_split['y_left'], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    right_subtree = build_tree(best_split['X_right'], best_split['y_right'], depth + 1, max_depth, min_samples_split, min_samples_leaf)
    return DecisionNode(feature=best_split['feature'], threshold=best_split['threshold'],
                        left=left_subtree, right=right_subtree)

# Function to predict a single sample
def predict_tree(tree, sample):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature] <= tree.threshold:
        return predict_tree(tree.left, sample)
    else:
        return predict_tree(tree.right, sample)

# Random Forest implementation
class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, max_features=None, min_samples_split=2, min_samples_leaf=1):
        self.n_trees = n_trees
        self.max_depth = max_depth or 10
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.trees = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Hitung jumlah maksimum fitur
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(n_features)) # Ubah ke int
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(n_features)) # Ubah ke int
        elif self.max_features is None:
            self.max_features = int(np.sqrt(n_features)) # Gunakan sqrt sebagai default

            for _ in range(self.n_trees):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y.iloc[indices]  # Use .iloc to access by position

                # Feature sampling
                feature_indices = np.random.choice(n_features, self.max_features, replace=False).astype(int)
                X_sample = X_sample[:, feature_indices]

                tree = build_tree(X_sample, y_sample, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf)
                tree.feature_indices = feature_indices  # Store sampled feature indices
                self.trees.append(tree)
    def predict(self, X):
        # Ensure X is a NumPy array
        X = np.array(X)  # Convert to NumPy array for compatibility
        tree_predictions = []
        for tree in self.trees:
            predictions = []
            for sample in X:
                try:
                    pred = predict_tree(tree, sample[tree.feature_indices])
                    predictions.append(pred)
                except Exception as e:
                    print(f"Error with sample: {sample}, using default value")
                    predictions.append(0)  # Ganti 0 dengan nilai default yang sesuai
                    raise e
            tree_predictions.append(predictions)

        # Majority voting
        tree_predictions = np.array(tree_predictions).T
        return [Counter(tree_predictions[i]).most_common(1)[0][0] for i in range(len(X))]

    # print("Feature indices (tree):", tree.feature_indices)
    # print("Feature indices dtype:", tree.feature_indices.dtype)
    # print("Sample shape:", sample.shape)
    # print("Sample values:", sample)

# Prepare dataset 
X = np.array(X_scaled)  # Use scaled data
y = y

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_train = np.array(X_train)
X_test = np.array(X_test)
# Train Random Forest
rf = RandomForest(n_trees=20, max_depth=10)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import pickle

pickle.dump(rf, open('rf_model_7742.pkl', 'wb'))