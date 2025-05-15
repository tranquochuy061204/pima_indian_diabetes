
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.impute import KNNImputer
import math

data = pd.read_csv("./diabetes.csv")
data = data.drop_duplicates()

data.head()


clean_data_zeros = data.copy()

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
clean_data_zeros[zero_cols] = clean_data_zeros[zero_cols].replace(0, np.nan)

imputer = KNNImputer(n_neighbors=5)
clean_data_zeros[zero_cols] = imputer.fit_transform(clean_data_zeros[zero_cols])

features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction']
outcome = clean_data_zeros.iloc[:, -1]

def cap_outliers(df, features):
    for feature in features:

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        df[feature] = df[feature].clip(lower=lower_limit, upper=upper_limit)

    return df

clean_data = cap_outliers(clean_data_zeros, features)


X = clean_data.drop('Outcome', axis=1).values
y = clean_data['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum([p * math.log2(p) for p in probs if p > 0])


def information_gain_ratio(X_column, y):
    percentiles = [25, 50, 75]
    thresholds = np.percentile(X_column, percentiles)

    best_gain_ratio = -1
    best_threshold = None
    ent_before = entropy(y)

    for threshold in thresholds:
        left_idx = X_column <= threshold
        right_idx = X_column > threshold

        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            continue

        left_y, right_y = y[left_idx], y[right_idx]

        ent_after = (len(left_y) / len(y)) * entropy(left_y) + (len(right_y) / len(y)) * entropy(right_y)
        info_gain = ent_before - ent_after

        p_left = len(left_y) / len(y)
        p_right = len(right_y) / len(y)
        split_info = - (p_left * math.log2(p_left) + p_right * math.log2(p_right)) if p_left > 0 and p_right > 0 else 1e-10

        gain_ratio = info_gain / split_info

        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_threshold = threshold

    return best_gain_ratio, best_threshold



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        
        self.threshold = threshold    
        self.left = left             
        self.right = right            
        self.value = value           

    def is_leaf_node(self):
        return self.value is not None


def build_tree(X, y, depth=0, max_depth=5):
    if len(np.unique(y)) == 1:
        return Node(value=int(y[0]))

    if len(y) == 0 or depth >= max_depth:
        return Node(value=int(np.round(np.mean(y))))

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature_index in range(X.shape[1]):
        gain_ratio, threshold = information_gain_ratio(X[:, feature_index], y)
        if gain_ratio > best_gain:
            best_gain = gain_ratio
            best_feature = feature_index
            best_threshold = threshold

    if best_gain == -1:
        return Node(value=int(np.round(np.mean(y))))

    left_idx = X[:, best_feature] <= best_threshold
    right_idx = X[:, best_feature] > best_threshold

    left = build_tree(X[left_idx], y[left_idx], depth + 1, max_depth)
    right = build_tree(X[right_idx], y[right_idx], depth + 1, max_depth)

    return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)


def predict_single(input_data, tree):
    node = tree
    while not node.is_leaf_node():
        if input_data[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right
    return node.value

def predict(X, tree):
    return np.array([predict_single(sample, tree) for sample in X])

tree = build_tree(X, y, max_depth=4)


import pickle

# Lưu model
with open("tree_model.pkl", "wb") as f:
    pickle.dump(tree, f)