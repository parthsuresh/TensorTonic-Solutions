import numpy as np

def build_expression_graph(leaves, operations):
    """
    Returns: node records in creation order and the final node ID
    """
    output = []

    id_to_val = {}
    
    for leaf in leaves:
        output.append({
            **leaf, 
            "grad": 0.0, 
            "op": "", 
            "parents": []
        })
        id_to_val[leaf["id"]] = leaf["data"]

    def compute_data(left_val, right_val, op):
        if op == "*":
            return left_val * right_val
        elif op == "+":
            return left_val + right_val
        else:
            raise ValueError("Unsupported operation!")
    
    for operation in operations:
        data = compute_data(id_to_val[operation["left"]], id_to_val[operation["right"]], operation["op"])
        id_to_val[operation["id"]] = data
        output.append(
            {
                "id": operation["id"],
                "data": data,
                "grad": 0.0,
                "op": operation["op"],
                "parents": [operation["left"], operation["right"]]
            }
        )

    return (output, output[-1]["id"])