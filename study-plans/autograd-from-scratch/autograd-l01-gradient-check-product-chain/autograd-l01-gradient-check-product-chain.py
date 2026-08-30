import numpy as np

def gradient_check_product_chain(a, b, c, f, h):
    """
    Returns: the loss, analytic gradients, numerical gradients, and maximum absolute disagreement
    """
    e = a * b + c
    L = e * f
    
    dL_da = b * f
    dL_db = a * f
    dL_dc = f
    dL_df = e
    analytical_gradients = np.array([dL_da, dL_db, dL_dc, dL_df], dtype=np.float64)
    
    dL_da_num = (((a + h) * b + c) * f - L) / h
    dL_db_num = ((a * (b + h) + c) * f - L) / h
    dL_dc_num = ((a * b + (c + h)) * f - L) / h
    dL_df_num = ((a * b + c) * (f + h) - L) / h
    numerical_gradients = np.array([dL_da_num, dL_db_num, dL_dc_num, dL_df_num], dtype=np.float64)

    max_disagreement = np.max(np.abs(analytical_gradients - numerical_gradients))
    
    return (L, list(analytical_gradients), list(numerical_gradients), float(max_disagreement))