# software/src/erp/models/linear.py
"""Dinamica lineal: el caso con solucion cerrada contra el que se mide el EKF.

`LinearDynamics` existe para una sola cosa, y es la obligacion de la fase P4:
un EKF al que se le da una dinamica lineal DEBE reducirse exactamente a un
filtro de Kalman, y eso se puede comparar contra la recursion escrita a mano.
Sin esto, la unica evidencia de que el EKF esta bien es que sus graficos
parecen razonables sobre un modelo que nadie puede resolver a mano.

Este modulo no importa mujoco y no debe hacerlo nunca: el test que lo usa
comprueba que `mujoco` no este siquiera en `sys.modules`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.types import Array

__all__ = ["LinearDynamics", "affine_predict"]


def affine_predict(
    A: np.ndarray,
    B: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    x_op: np.ndarray,
    u_op: np.ndarray,
    x_next_op: np.ndarray,
) -> np.ndarray:
    """Modelo linealizado en DESVIACIONES respecto a (x_op, u_op)."""
    return np.asarray(x_next_op + A @ (x - x_op) + B @ (u - u_op), dtype=np.float64)


class LinearDynamics:
    """`x' = A x + B u`, `z = C x`. Cumple `DiscreteDynamics`.

    Las jacobianas son constantes: `step` devuelve `A` y `observe` devuelve `C`
    sin mirar `x`. Eso es justamente lo que hace que un EKF sobre esta dinamica
    sea un KF -- la linealizacion no aproxima nada porque no hay nada que
    aproximar -- y por eso la comparacion contra la recursion cerrada es una
    igualdad y no un "parecido".

    Parameters
    ----------
    A: (nx, nx)   B: (nx, nu)   C: (nz, nx)
    dt:
        Paso nominal. No se usa en la aritmetica; esta para cumplir el
        protocolo y para que quien arme una `Q` sepa contra que paso la armo.
    """

    def __init__(
        self,
        A: npt.ArrayLike,
        B: npt.ArrayLike | None = None,
        C: npt.ArrayLike | None = None,
        *,
        dt: float = 1.0,
    ) -> None:
        self.A: Array = np.atleast_2d(np.asarray(A, dtype=np.float64))
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError(f"A tiene que ser cuadrada, es {self.A.shape}")
        self.nx: int = int(self.A.shape[0])
        self.B: Array = (
            np.zeros((self.nx, 0)) if B is None else np.atleast_2d(np.asarray(B, dtype=np.float64))
        )
        if self.B.shape[0] != self.nx:
            raise ValueError(f"B tiene que tener {self.nx} filas, tiene {self.B.shape[0]}")
        self.C: Array = (
            np.eye(self.nx) if C is None else np.atleast_2d(np.asarray(C, dtype=np.float64))
        )
        if self.C.shape[1] != self.nx:
            raise ValueError(f"C tiene que tener {self.nx} columnas, tiene {self.C.shape[1]}")
        self.nz: int = int(self.C.shape[0])
        self.nu: int = int(self.B.shape[1])
        self.dt: float = float(dt)

    def step(self, x: npt.ArrayLike, u: npt.ArrayLike) -> tuple[Array, Array]:
        """`(A x + B u, A)`. La jacobiana no depende de `x`."""
        x_arr = np.asarray(x, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64)
        x_next = self.A @ x_arr + (self.B @ u_arr if self.nu else 0.0)
        return np.asarray(x_next, dtype=np.float64), self.A

    def observe(self, x: npt.ArrayLike, u: npt.ArrayLike) -> tuple[Array, Array]:
        """`(C x, C)`. `u` se acepta y se ignora: no hay termino directo."""
        z = self.C @ np.asarray(x, dtype=np.float64)
        return np.asarray(z, dtype=np.float64), self.C

    def __repr__(self) -> str:
        return f"LinearDynamics(nx={self.nx}, nz={self.nz}, nu={self.nu}, dt={self.dt})"
