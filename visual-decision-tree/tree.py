import numpy as np
import pandas as pd
import copy
from collections import Counter
import json

class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, samples_idx=None, class_name=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # numeric class ID at leaf
        self.class_name = class_name  # actual label name (string)
        self.samples_idx = samples_idx  # indices of samples reaching this node (for visualization/counts)

    def is_leaf(self):
        return self.value is not None

    def to_dict(self):
        if self.is_leaf():
            return {
                "leaf": True,
                "value": int(self.value) if self.value is not None else None,
                "class_name": str(self.class_name) if self.class_name is not None else None,
                "samples": len(self.samples_idx) if self.samples_idx is not None else None
            }
        else:
            return {
                "leaf": False,
                "feature": self.feature,
                "threshold": None if self.threshold is None else float(self.threshold) if isinstance(self.threshold, (int, float)) else str(self.threshold),
                "left": self.left.to_dict() if self.left else None,
                "right": self.right.to_dict() if self.right else None,
                "samples": len(self.samples_idx) if self.samples_idx is not None else None
            }

#Impurity functions
def entropy(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

def gini(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return 1.0 - np.sum(probs ** 2)

def all_same(y):
    return np.unique(y).size == 1

def majority_class(y):
    vals, counts = np.unique(y, return_counts=True)
    return vals[np.argmax(counts)]

def possible_splits_numeric(col_values):
    vals = np.unique(np.sort(col_values))
    if len(vals) <= 1:
        return []
    thresholds = (vals[:-1] + vals[1:]) / 2.0
    return thresholds.tolist()

#Find best split
def find_best_split(X_df, y_arr, feature_names, impurity='gini', min_samples_split=2):
    best_feature = None
    best_threshold = None
    best_score = np.inf
    best_info = None

    impurity_fn = gini if impurity == 'gini' else entropy
    n = len(y_arr)
    if n < min_samples_split:
        return None, None, None

    for feature in feature_names:
        col = X_df[feature]
        if pd.api.types.is_numeric_dtype(col):
            thresholds = possible_splits_numeric(col.values)
            for thr in thresholds:
                left_idx = col.values <= thr
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                left_imp = impurity_fn(y_arr[left_idx])
                right_imp = impurity_fn(y_arr[right_idx])
                score = (left_idx.sum() / n) * left_imp + (right_idx.sum() / n) * right_imp
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = thr
                    best_info = {
                        "score": score,
                        "left_count": int(left_idx.sum()),
                        "right_count": int(right_idx.sum()),
                        "left_impurity": float(left_imp),
                        "right_impurity": float(right_imp)
                    }
        else:
            # categorical: simple one-vs-rest splits for each category
            cats = col.astype('str').unique()
            for cat in cats:
                left_idx = col.astype('str') == str(cat)
                right_idx = ~left_idx
                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue
                left_imp = impurity_fn(y_arr[left_idx])
                right_imp = impurity_fn(y_arr[right_idx])
                score = (left_idx.sum() / n) * left_imp + (right_idx.sum() / n) * right_imp
                if score < best_score:
                    best_score = score
                    best_feature = feature
                    best_threshold = cat
                    best_info = {
                        "score": score,
                        "left_count": int(left_idx.sum()),
                        "right_count": int(right_idx.sum()),
                        "left_impurity": float(left_imp),
                        "right_impurity": float(right_imp)
                    }

    return best_feature, best_threshold, best_info

# Split dataset 
def split_dataset(X_df, y_arr, best_feature, best_threshold):
    # Handle numeric and categorical separately
    if pd.api.types.is_numeric_dtype(X_df[best_feature]):
        left_mask = X_df[best_feature] <= best_threshold
        right_mask = X_df[best_feature] > best_threshold
    else:
        left_mask = X_df[best_feature].astype(str) == str(best_threshold)
        right_mask = ~left_mask

    X_left = X_df[left_mask].reset_index(drop=True)
    y_left = y_arr[left_mask]
    X_right = X_df[right_mask].reset_index(drop=True)
    y_right = y_arr[right_mask]

    return X_left, y_left, X_right, y_right, left_mask, right_mask

# Build tree trace 
def build_tree_trace(X_df, y_arr, feature_names, impurity='gini', max_depth=None, min_samples_split=2, class_names=None):
    """
    Generator that yields (root_snapshot, step_info) after each split.
    Automatically maps numeric y_arr to class_names if provided.
    """
    # If y_arr is categorical or string, encode it but remember mapping
    if class_names is None:
        if y_arr.dtype == 'object' or y_arr.dtype.name == 'category':
            classes = pd.Categorical(y_arr).categories.tolist()
            y_encoded = pd.Categorical(y_arr, categories=classes).codes
            class_names = classes
        else:
            unique_classes = np.unique(y_arr)
            class_names = [str(c) for c in unique_classes]
            y_encoded = y_arr
    else:
        y_encoded = y_arr

    node_counter = {"id": 0}
    steps = []

    def build_and_record(X_sub, y_sub, depth):
        node_counter["id"] += 1
        nid = node_counter["id"]
        samples_idx = np.arange(len(y_sub))
        if len(y_sub) == 0 or all_same(y_sub) or (max_depth is not None and depth >= max_depth) or len(y_sub) < min_samples_split:
            maj_class = int(majority_class(y_sub)) if len(y_sub) > 0 else None
            maj_label = class_names[maj_class] if maj_class is not None and maj_class < len(class_names) else None
            return DecisionTreeNode(value=maj_class, class_name=maj_label, samples_idx=samples_idx)

        bf, bt, info = find_best_split(X_sub, y_sub, feature_names, impurity, min_samples_split)
        if bf is None:
            maj_class = int(majority_class(y_sub))
            maj_label = class_names[maj_class]
            return DecisionTreeNode(value=maj_class, class_name=maj_label, samples_idx=samples_idx)

        node = DecisionTreeNode(feature=bf, threshold=bt, samples_idx=samples_idx)
        steps.append({
            "node_id": nid,
            "depth": depth,
            "chosen_feature": bf,
            "threshold": bt,
            "split_info": info,
            "n_samples": len(y_sub),
            "class_counts": {class_names[k]: int(v) for k, v in Counter(y_sub.tolist()).items()}
        })
        X_left, y_left, X_right, y_right, _, _ = split_dataset(X_sub, y_sub, bf, bt)
        node.left = build_and_record(X_left, y_left, depth + 1)
        node.right = build_and_record(X_right, y_right, depth + 1)
        return node

    root = build_and_record(X_df.copy().reset_index(drop=True), y_encoded.copy(), 0)
    snapshots = []
    for idx, step in enumerate(steps):
        snapshots.append({
            "root": copy.deepcopy(root),
            "active_step_index": idx,
            "step": step,
            "all_steps": steps
        })

    if len(snapshots) == 0:
        snapshots.append({
            "root": root,
            "active_step_index": None,
            "step": None,
            "all_steps": []
        })

    for snap in snapshots:
        yield snap

#Prediction
def predict_single(root: DecisionTreeNode, x_row: pd.Series):
    node = root
    while not node.is_leaf():
        feat = node.feature
        thr = node.threshold
        val = x_row[feat]
        if pd.api.types.is_numeric_dtype(type(val)) or isinstance(val, (int, float, np.number)):
            go_left = val <= float(thr)
        else:
            go_left = str(val) == str(thr)
        node = node.left if go_left else node.right
    return node.class_name if node.class_name else node.value

# JSON Export 
def tree_to_json(root: DecisionTreeNode):
    return json.dumps(root.to_dict(), indent=2)
