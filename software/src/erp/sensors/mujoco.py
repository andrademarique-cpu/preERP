# software/src/erp/sensors/mujoco.py
import numpy as np
import mujoco as mj

def sensor_slice(name: str, model: mj.MjModel) -> slice:
    """Returns the slice of sensordata (and H) occupied by a specific sensor."""
    s = model.sensor(name)
    adr, dim = int(s.adr[0]), int(s.dim[0])
    return slice(adr, adr + dim)

def rows_of(model: mj.MjModel, *names: str) -> np.ndarray:
    """Returns the concatenated sensordata indices for a set of sensors.
    
    Useful for indexing the measurement vector z, covariance R, 
    and the rows of the observation Jacobian H.
    """
    sl = [sensor_slice(n, model) for n in names]
    return np.concatenate([np.arange(s.start, s.stop) for s in sl])
    
def make_R(model: mj.MjModel, sig_acc: float, sig_gyr: float) -> np.ndarray:
    """Diagonal R matrix based on sensor specs."""
    R = np.zeros((model.nsensordata,) * 2)
    for i in range(model.nsensor):
        nm = model.sensor(i).name
        sl = sensor_slice(nm, model)
        s = sig_acc if "acc" in nm else (sig_gyr if "gyro" in nm else 1.0)
        for k in range(sl.start, sl.stop):
            R[k, k] = s ** 2
    return R

def sensor_noise(Z_true: np.ndarray, seed: int, rows: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Injects white Gaussian noise into truth sensor data."""
    sd = np.sqrt(np.diag(R))
    rng = np.random.default_rng(seed)
    Z = np.array(Z_true, dtype=float, copy=True)
    Z[:, rows] += rng.normal(0, sd[rows], size=(len(Z), len(rows)))
    return Z