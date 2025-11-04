# viz.py
from graphviz import Digraph
from collections import Counter
import numpy as np

def node_label(node, show_counts=True):
    if node is None:
        return ""
    if node.is_leaf():
        class_display = node.class_name if node.class_name is not None else node.value
        txt = f"Leaf\\nClass: {class_display}"

    else:
        thr = node.threshold
        txt = f"{node.feature}\\n<= {thr}"
    if show_counts and node.samples_idx is not None:
        txt += f"\\nN={len(node.samples_idx)}"
    if show_counts and node.is_leaf() and node.samples_idx is not None:
        # we can't get counts properly here since samples_idx are local, but keep simple
        pass
    return txt

def build_dot(root, active_step=None, show_counts=True):
    dot = Digraph(node_attr={'shape':'box','style':'rounded,filled','fontname':'Helvetica'})
    counter = {"nid":0}
    def add_nodes(node):
        if node is None:
            return None
        counter["nid"] += 1
        nid = f"n{counter['nid']}"
        label = node_label(node, show_counts=show_counts)
        # highlight active node if matches
        if active_step is not None and not node.is_leaf() and node.feature == active_step.get("chosen_feature") and str(node.threshold) == str(active_step.get("threshold")):
            dot.node(nid, label, fillcolor='lightgoldenrod', color='darkgoldenrod')
        else:
            fill = 'white' if node.is_leaf() else 'lightgrey'
            dot.node(nid, label, fillcolor=fill)
        # store id for edges via attributes
        node._viz_id = nid
        if node.left:
            add_nodes(node.left)
            dot.edge(nid, node.left._viz_id, label="True")
        if node.right:
            add_nodes(node.right)
            dot.edge(nid, node.right._viz_id, label="False")
        return nid
    add_nodes(root)
    return dot.source  # DOT string

def dot_to_png_bytes(dot_source):
    g = Digraph(comment='tree')
    g.source = dot_source
    return g.pipe(format='png')
