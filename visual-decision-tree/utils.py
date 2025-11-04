# utils.py
import pandas as pd
import numpy as np
from io import BytesIO
import json
from tree import tree_to_json

def load_csv(file_like):
    df = pd.read_csv(file_like)
    return df

def df_overview(df):
    overview = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": []
    }
    for c in df.columns:
        overview["columns"].append({
            "name": c,
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True))
        })
    return overview

def export_tree_json(root):
    return tree_to_json(root)

def export_python_predict_function(root, feature_order=None, function_name="predict_tree"):
    # Very simple generator: uses the tree recursively to create nested ifs
    def node_to_code(node, indent=4):
        sp = " " * indent
        if node.is_leaf():
            return sp + f"return {int(node.value)}\n"
        thr = node.threshold
        cond = f"x['{node.feature}'] <= {thr}" if isinstance(thr, (int,float)) else f"str(x['{node.feature}']) == '{thr}'"
        code = sp + f"if {cond}:\n"
        code += node_to_code(node.left, indent+4)
        code += sp + "else:\n"
        code += node_to_code(node.right, indent+4)
        return code

    code = f"def {function_name}(x):\n"
    code += node_to_code(root, indent=4)
    return code
