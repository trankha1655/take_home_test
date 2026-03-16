from prettytable import PrettyTable 
from sty import fg, bg, ef, rs
import copy
import numpy as np

_all__ = [
    "pretty_output",
    "padding_str",
]


def pretty_output(label_list, metrics, fg_color=40, name="Class Name"):
    metric_name = ["Acc", "Recall" ,"Precision", "F1"]
    myTable = PrettyTable([name, *metric_name])

    metrics = copy.deepcopy(metrics)
    for k, v in metrics.items():
        metrics[k] = np.round_(v, decimals = 3) 

    for i, label_name in enumerate(label_list):
        row = [label_name] + [metrics[name.lower()][i] for name in  metric_name]
        myTable.add_row(row)

    lines = myTable.get_string().split("\n")
    for line in lines:
        output = fg(fg_color) + line + fg.rs
        print(output)

def padding_str(name, n_max):
    nn = len(name)
    n = n_max - nn
    if n <= 0:
        return name
    
    name = ' '* (n//2) + name + ' ' * (n-n//2)
    return name

