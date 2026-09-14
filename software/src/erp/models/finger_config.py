# software/src/erp/models/finger_config.py
from dataclasses import dataclass, field
from typing import Dict, Tuple

@dataclass
class FingerGeometry:
    l0: float = 0.030
    w0: float = 0.008
    l1: float = 0.035
    w1: float = 0.006
    i1: float = 0.0175
    l2: float = 0.030
    w2: float = 0.005
    i2: float = 0.0150
    b: float = 0.002

@dataclass
class SimOptions:
    timestep: float = 0.002
    gravity: str = "0 0 -9.81"
    integrator: str = "Euler"

@dataclass
class ActuatorSpecs:
    joint: str
    kp: float
    kv: float
    force_max: float
    armature: float
    damping: float
    tau: float

@dataclass
class FingerRobotConfig:
    geometry: FingerGeometry = field(default_factory=FingerGeometry)
    sim_options: SimOptions = field(default_factory=SimOptions)
    
    joints: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=lambda: {
        "joint_1": {"range_deg": (-90.0, 90.0)},
        "joint_2": {"range_deg": (-120.0, 120.0)},
    })
    
    actuators: Dict[str, ActuatorSpecs] = field(default_factory=lambda: {
        "act_joint_1": ActuatorSpecs(
            joint="joint_1", kp=2.0, kv=0.025, force_max=0.30, armature=4e-5, damping=1e-3, tau=0.004
        ),
        "act_joint_2": ActuatorSpecs(
            joint="joint_2", kp=0.8, kv=0.010, force_max=0.15, armature=2e-5, damping=8e-4, tau=0.004
        ),
    })