# software/src/erp/sim/mujoco.py
import numpy as np
import mujoco as mj

def state_dim(m: mj.MjModel) -> int:
    """nx de MuJoCo: posiciones en el espacio tangente (nv, no nq) + activaciones."""
    return 2 * m.nv + m.na

def _load(model: mj.MjModel, data: mj.MjData, x: np.ndarray, u: np.ndarray) -> None:
    """Estado del filtro -> MjData. mj_resetData primero para evitar estado residual."""
    nv, na = model.nv, model.na
    mj.mj_resetData(model, data)
    data.qpos[:nv] = x[:nv]
    data.qvel[:nv] = x[nv:2 * nv]
    if na:
        data.act[:na] = x[2 * nv:2 * nv + na]
    data.ctrl[:] = u

def _dump(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """MjData -> vector de estado, en el mismo orden que espera _load."""
    nv, na = model.nv, model.na
    p = [data.qpos[:nv].copy(), data.qvel[:nv].copy()]
    if na:
        p.append(data.act[:na].copy())
    return np.concatenate(p)

def f_dyn(x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """x_{k+1} = f(x_k, u_k) -- un mj_step. -> (nx,)"""
    _load(model, data, x, u)
    mj.mj_step(model, data)
    return _dump(model, data)

def F_dyn(x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6) -> np.ndarray:
    """F = df/dx evaluado en (x, u) -- el bloque A de mjd_transitionFD."""
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    n = state_dim(model)
    F = np.zeros((n, n))
    mj.mjd_transitionFD(model, data, eps, True, F, None, None, None)
    return F

def h_dyn(x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """z = h(x, u) -- que leerian los sensores en el estado x con el control u."""
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    return data.sensordata.copy()

def H_dyn(x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6) -> np.ndarray:
    """H = dh/dx -- el bloque C de mjd_transitionFD."""
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    H = np.zeros((model.nsensordata, state_dim(model)))
    mj.mjd_transitionFD(model, data, eps, True, None, None, H, None)
    return H

def linearize(model: mj.MjModel, data: mj.MjData, x_op: np.ndarray, u_op: np.ndarray, eps: float = 1e-6, centered: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A, B, C, D en el punto de operacion (x_op, u_op)."""
    _load(model, data, x_op, u_op)
    mj.mj_forward(model, data)
    n, nu, ns = state_dim(model), model.nu, model.nsensordata
    
    A = np.zeros((n, n))
    B = np.zeros((n, nu))
    C = np.zeros((ns, n))
    D = np.zeros((ns, nu))
    mj.mjd_transitionFD(model, data, eps, centered, A, B, C, D)
    return A, B, C, D

def make_Q(model: mj.MjModel, sig_alpha: float, sig_act: float, sig_cm: float = 0.0, dt: float = None) -> np.ndarray:
    """Q matrix for discrete white noise acceleration (DWNA)."""
    dt = model.opt.timestep if dt is None else dt
    nv, na, nx = model.nv, model.na, state_dim(model)
    G = np.zeros((nx, nv))
    for j in range(nv):
        G[j, j] = dt ** 2 / 2
        G[nv + j, j] = dt
        
    Q = G @ (sig_alpha ** 2 * np.eye(nv)) @ G.T
    for j in range(na):
        Q[2 * nv + j, 2 * nv + j] = sig_act ** 2
        
    for j in range(na):
        e = np.zeros(nx)
        e[j] = 1.0              # q_j
        e[2 * nv + j] = 1.0     # a_j, same sign -> force kp*(a-q) invariant
        Q = Q + sig_cm ** 2 * np.outer(e, e)
        
    return Q + np.eye(nx) * 1e-18