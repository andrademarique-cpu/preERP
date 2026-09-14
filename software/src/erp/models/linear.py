# software/src/erp/models/linear.py
import numpy as np

def affine_predict(A: np.ndarray, B: np.ndarray, x: np.ndarray, u: np.ndarray, x_op: np.ndarray, u_op: np.ndarray, x_next_op: np.ndarray) -> np.ndarray:
    """Modelo linealizado en DESVIACIONES respecto a (x_op, u_op)."""
    return x_next_op + A @ (x - x_op) + B @ (u - u_op)