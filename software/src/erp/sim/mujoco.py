# software/src/erp/sim/mujoco.py
import mujoco as mj
import numpy as np


def state_dim(m: mj.MjModel) -> int:
    """nx de MuJoCo: posiciones en el espacio tangente (nv, no nq) + activaciones."""
    return int(2 * m.nv + m.na)

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

def F_dyn(
    x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6
) -> np.ndarray:
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
    # Annotated and copied: data.sensordata is Any (mujoco has no py.typed) and
    # is a live view into MjData that the next mj_forward overwrites.
    z: np.ndarray = np.array(data.sensordata, dtype=np.float64)
    return z

def H_dyn(
    x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6
) -> np.ndarray:
    """H = dh/dx -- el bloque C de mjd_transitionFD."""
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    H = np.zeros((model.nsensordata, state_dim(model)))
    mj.mjd_transitionFD(model, data, eps, True, None, None, H, None)
    return H

def step_with_jacobian(
    x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(x_{k+1}, F) desde UNA sola carga de MjData, en vez de dos.

    Equivale exactamente a `F_dyn(x, u)` seguido de `f_dyn(x, u)` -- verificado
    bit a bit, no `allclose`. Se puede compartir la carga porque
    `mjd_transitionFD` deja `qpos`/`qvel`/`act` como los encontro, asi que el
    `mj_step` de abajo arranca del estado nominal.

    Devolver el par junta las dos cosas que TIENEN que salir del mismo punto de
    linealizacion: F es df/dx evaluada en el x de ANTES del paso. Pedirlas por
    separado deja abierta la posibilidad de evaluarlas en x distintos, que es un
    bug que no se ve -- el filtro sigue corriendo y solo diverge de a poco.

    El ahorro de tiempo es real pero chico (~11% del predict): el costo lo pone
    `mjd_transitionFD`, que hace nx+1 evaluaciones internas, no la carga.
    """
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    n = state_dim(model)
    F = np.zeros((n, n))
    mj.mjd_transitionFD(model, data, eps, True, F, None, None, None)
    mj.mj_step(model, data)
    return _dump(model, data), F


def observe_with_jacobian(
    x: np.ndarray, u: np.ndarray, model: mj.MjModel, data: mj.MjData, eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """(z, H) desde UNA sola carga. Equivale a `H_dyn(x, u)` + `h_dyn(x, u)`.

    OJO CON EL SEGUNDO `mj_forward`, QUE NO ES REDUNDANTE. `mjd_transitionFD`
    restaura el ESTADO pero deja `sensordata` con lo que calculo en su ultima
    perturbacion. Leerlo directo despues da un z equivocado en 4.3e-4 -- contra
    un sig_acc de 0.05 es ~1% de sigma: chico como para parecer un problema de
    tolerancia y grande como para mover la corrida congelada. El `mj_forward`
    recalcula los sensores desde el estado ya restaurado.
    """
    _load(model, data, x, u)
    mj.mj_forward(model, data)
    H = np.zeros((model.nsensordata, state_dim(model)))
    mj.mjd_transitionFD(model, data, eps, True, None, None, H, None)
    mj.mj_forward(model, data)
    z: np.ndarray = np.array(data.sensordata, dtype=np.float64)
    return z, H


def linearize(
    model: mj.MjModel,
    data: mj.MjData,
    x_op: np.ndarray,
    u_op: np.ndarray,
    eps: float = 1e-6,
    centered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

def make_Q(
    model: mj.MjModel,
    sig_alpha: float,
    sig_act: float,
    sig_cm: float = 0.0,
    dt: float | None = None,
) -> np.ndarray:
    """Q matrix for discrete white noise acceleration (DWNA).

    `dt` defaults to `model.opt.timestep`. DWNA is NOT schedule invariant, so
    the result is only valid at the step it was built for.
    """
    dt_s = float(model.opt.timestep) if dt is None else float(dt)
    nv, na, nx = model.nv, model.na, state_dim(model)
    G = np.zeros((nx, nv))
    for j in range(nv):
        G[j, j] = dt_s ** 2 / 2
        G[nv + j, j] = dt_s


    Q = G @ (sig_alpha ** 2 * np.eye(nv)) @ G.T
    for j in range(na):
        Q[2 * nv + j, 2 * nv + j] = sig_act ** 2
        
    for j in range(na):
        e = np.zeros(nx)
        e[j] = 1.0              # q_j
        e[2 * nv + j] = 1.0     # a_j, same sign -> force kp*(a-q) invariant
        Q = Q + sig_cm ** 2 * np.outer(e, e)
        
    return np.asarray(Q + np.eye(nx) * 1e-18, dtype=np.float64)