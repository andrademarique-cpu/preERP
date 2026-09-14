# software/src/erp/estimators/ekf.py
import mujoco as mj
import numpy as np

from erp.core.linalg import make_spd
from erp.sim.mujoco import F_dyn, H_dyn, f_dyn, h_dyn


class EKF:
    """EKF con f/F/h/H de MuJoCo. Update en forma de Joseph. CIEGO al control.

    El filtro nunca recibe u. `self.u_blind` es el UNICO ctrl que escribe -- se
    aloca una vez, en cero, y va a f, F, h y H por igual, asi que las cuatro
    funciones dependen solo del estado. No hay forma de pasarle un control: los
    metodos no tienen el parametro.

    Que eso sea sano o no depende enteramente del modelo que se le pase:
    model_blind deja la activacion como random walk, model_sim la decae a cero.

    x0 : (nx,) estado inicial       P0 : (nx,nx) covarianza inicial
    Q  : (nx,nx) ruido de proceso   R  : (ns,ns) ruido de medicion, sensordata COMPLETA
    """

    def __init__(self, x0: np.ndarray, P0: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 model: mj.MjModel, data: mj.MjData) -> None:
        self.x = np.asarray(x0, float).copy()
        self.P = make_spd(np.asarray(P0, float))
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.model, self.data = model, data
        self.u_blind = np.zeros(model.nu)   # el unico ctrl que ve el filtro

    def predict(self) -> None:
        """t_k -> t_{k+1}, un paso de model.opt.timestep."""
        u = self.u_blind
        F = F_dyn(self.x, u, self.model, self.data)
        self.x = f_dyn(self.x, u, self.model, self.data)
        self.P = make_spd(F @ self.P @ F.T + self.Q)

    def update(self, z: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, float]:
        """Corrige con los canales `rows` de la medicion z (sensordata completa).
        -> (innovacion, NIS).

        solve y no inv: S se pone mal condicionada cuando el dedo se estira.
        Joseph y no (I-KH)P: sobrevive el redondeo que la forma corta no.
        """
        u = self.u_blind
        H = H_dyn(self.x, u, self.model, self.data)[rows]
        y = z[rows] - h_dyn(self.x, u, self.model, self.data)[rows]
        R = self.R[np.ix_(rows, rows)]
        S = make_spd(H @ self.P @ H.T + R)
        K = np.linalg.solve(S, H @ self.P).T          # = P H^T S^-1
        self.x = self.x + K @ y
        I_KH = np.eye(self.x.size) - K @ H
        self.P = make_spd(I_KH @ self.P @ I_KH.T + K @ R @ K.T)
        return y, float(y @ np.linalg.solve(S, y))
