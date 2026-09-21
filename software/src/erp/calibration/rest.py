"""Static calibration from a window in which the arm was not moving.

**Import direction.** This module depends on :mod:`erp.core.types` and numpy,
and on nothing else in ``erp``. In particular it must never import
:mod:`erp.sensors`: ``sensors/`` imports *this*, not the reverse. CLAUDE.md
names "importing a ``calibration/`` module that itself imports ``sensors/``" as
one of the reach-throughs the CI grep cannot see, so the direction is kept
structurally impossible rather than merely agreed.

English, beside ``sensors/`` and ``robot/``: this is sensor plumbing, not
estimation. `CODEOWNERS` groups it the same way.

Two entry points, one arithmetic:

- :func:`calibration_from_samples` takes samples already selected and is what
  a live :class:`~erp.sensors.imu_serial.SerialIMUSensor` calls from
  ``calibrate()``;
- :func:`rest_bias` takes a whole logged run and selects the still window out
  of it, which is what the offline pipeline needs.

What a rest calibration can and cannot do
-----------------------------------------
At rest a gyro should read 0 and an accelerometer reads **gravity**, not zero,
so the accelerometer bias is only estimable against an ``expected_rest`` from
the model -- ``h(x_rest)`` at the sensor rows. Without one, only the gyro bias
is touched and the result says so in its ``note``.

The result is valid *for that pose only*. A mounting tilt is not an additive
offset once the link rotates, so what this removes is "bias + mounting at this
pose", and calling it a sensor bias is a convenience. `docs/theory/
finger_imu_ekf.md` § 6 is the standing warning that inflating or re-fitting
``R`` does not repair a bias.

On the ``R`` this returns
-------------------------
It is the sample covariance of a **static** window: the noise floor at rest,
not the noise in motion, and it carries nothing about model error. Measured on
the golden run's 8-sample window it is 3-8x tighter than the nominal
``make_R`` block, so adopting it would make an already over-confident filter
(NIS median 29 against a target of 12) considerably worse. It is computed and
returned; the pipeline deliberately does not use it. See ADR-0002's P6 record.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from erp.core.types import CalibrationResult

__all__ = ["calibration_from_samples", "rest_bias"]


def calibration_from_samples(
    Z: npt.ArrayLike,
    current_bias: npt.ArrayLike,
    *,
    gyro_idx: npt.ArrayLike,
    acc_idx: npt.ArrayLike,
    expected_rest: npt.ArrayLike | None = None,
    still_gyro_std: float = 0.02,
    min_samples: int = 20,
) -> CalibrationResult:
    """Static calibration from samples taken while the arm is not moving.

    Parameters
    ----------
    Z:
        (N, dim) calibrated samples, produced with ``current_bias`` applied.
    current_bias:
        (dim,) bias in effect while ``Z`` was recorded.
    gyro_idx, acc_idx:
        Positions of gyro (rad/s) and accelerometer (m/s^2) channels in ``z``.
    expected_rest:
        (dim,) what MuJoCo's sensors read at the rest pose, e.g. ``h(x_rest)``
        at the rows. Required to estimate accelerometer bias: at rest an
        accelerometer reads gravity, not zero.

        **When given it applies to every channel, gyro included** -- the bias
        is ``mean(Z) - expected_rest`` throughout. Passing zeros for the gyro
        entries therefore reproduces the older behaviour exactly, which is what
        every existing caller does. The difference matters for one caller: the
        offline pipeline passes ``h(x_rest)``, whose gyro entries are **not**
        zero but up to 2.4e-4 rad/s, because ``warmup_to_rest`` leaves a little
        residual velocity after its 100 steps. That is ~4.7% of ``sig_gyro``.

        Which is correct is a real question and P6 answered it conservatively:
        a resting gyro *should* read 0, so ``h(x_rest)[gyro]`` is a model
        artifact being folded into a sensor bias, and expecting 0 is the better
        physics. It is not adopted because doing so moves the frozen golden run
        -- a deliberate act, not a refactor. To get that behaviour, zero the
        gyro entries of ``expected_rest`` before passing it.
    still_gyro_std:
        rad/s. A gyro channel noisier than this means the arm moved, and the
        result is marked invalid instead of baking motion into the bias.
    min_samples:
        Fewer samples than this -> invalid. The default suits a live 2 s
        calibration at 20 Hz (~40 samples). It is **too high for the offline
        rest window**, which is 0.4 s and therefore 8 samples -- see
        :func:`rest_bias`, which lowers it deliberately.

    Returns
    -------
    A :class:`CalibrationResult` whose ``bias`` is the *total* bias to use
    (current + correction) and whose ``R`` is the sample covariance. ``bias``
    is meant to be **subtracted**, matching ``IMUDecoder.apply``, which
    computes ``raw @ axis_map - b``.
    """
    Z_a = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    b0 = np.asarray(current_bias, dtype=np.float64)
    dim = b0.size
    g = np.asarray(gyro_idx, dtype=np.intp)
    a = np.asarray(acc_idx, dtype=np.intp)
    # `acc_idx` no longer selects which channels get a correction -- with an
    # `expected_rest` every channel does -- so it is checked instead of used.
    # Overlapping gyro and accelerometer indices means the caller built them
    # from mismatched layouts, and the result would be a plausible-looking
    # bias computed from the wrong channels.
    if np.intersect1d(g, a).size:
        raise ValueError(
            f"gyro_idx and acc_idx overlap at {np.intersect1d(g, a).tolist()}; "
            "they are positions in the same z vector and cannot be the same channel"
        )
    n = Z_a.shape[0] if Z_a.size else 0

    if n < min_samples:
        return CalibrationResult(
            bias=b0.copy(), scale=np.ones(dim), R=np.zeros((dim, dim)), valid=False,
            note=f"only {n} samples, need {min_samples}",
        )

    mean = Z_a.mean(axis=0)
    R = np.atleast_2d(np.cov(Z_a, rowvar=False)) + 1e-12 * np.eye(dim)
    delta = np.zeros(dim)
    notes = [f"{n} samples"]
    if expected_rest is not None:
        # Every channel, not just `a`. See the parameter's docstring: the gyro
        # entries are zero for every caller but the offline pipeline, so this
        # is a no-op for them and is what makes the offline path reproduce the
        # notebook cell bit-for-bit.
        delta = mean - np.asarray(expected_rest, dtype=np.float64)
    else:
        delta[g] = mean[g]
        notes.append("accelerometer bias not estimated (no expected_rest)")

    gyro_std = float(Z_a[:, g].std(axis=0).max()) if g.size else 0.0
    valid = gyro_std <= still_gyro_std
    if not valid:
        notes.append(f"gyro std {gyro_std:.4f} rad/s > {still_gyro_std} -- arm moved?")
    return CalibrationResult(
        bias=b0 + delta, scale=np.ones(dim), R=R, valid=valid, note="; ".join(notes)
    )


def rest_bias(
    t: npt.ArrayLike,
    Z: npt.ArrayLike,
    *,
    expected_rest: npt.ArrayLike,
    gyro_idx: npt.ArrayLike,
    acc_idx: npt.ArrayLike,
    rest_s: float = 0.4,
    max_gyro_norm: float = 0.05,
    min_samples: int = 3,
    still_gyro_std: float = 0.02,
    current_bias: npt.ArrayLike | None = None,
) -> CalibrationResult:
    """Calibration from the still window at the START of a logged run.

    The offline counterpart of
    :meth:`~erp.sensors.imu_serial.SerialIMUSensor.calibrate`: the live sensor
    is told *when* to hold still, a recorded run has to be told where to look.

    Returns a result rather than raising, which is the whole point of the
    ``valid`` / ``note`` shape -- the caller decides whether an unusable
    calibration is fatal. The offline pipeline treats it as fatal, because
    proceeding would silently subtract a bias of zero and quietly change what
    the run means.

    Parameters
    ----------
    t:
        (N,) sample times in seconds, relative to the start of the run.
    Z:
        (N, dim) calibrated samples, in the units and frame of ``expected_rest``.
    expected_rest:
        (dim,) ``h(x_rest)`` at the sensor rows. See
        :func:`calibration_from_samples` for what its gyro entries mean.
    rest_s:
        The window is ``t < rest_s``. The arm starts ~0.5 s late on this rig,
        so 0.4 s of genuinely still data is available; at 0.6 s the window
        already catches the start of the motion and the guard below fires.
    max_gyro_norm:
        rad/s. If the largest gyro-vector norm in the window exceeds this, the
        arm was moving and the result is invalid.
    min_samples:
        Defaults to **3**, not to the 20 of
        :func:`calibration_from_samples`. At 0.4 s and a ~20 Hz IMU the window
        holds 8 samples, so the live default would reject every offline run --
        and reject it by returning a zero bias, which looks like a successful
        calibration to anything that does not check ``valid``.
    still_gyro_std:
        Passed through. Both guards apply: the norm catches a slow sweep whose
        per-channel scatter stays small, the standard deviation catches jitter
        whose mean cancels. On the golden run's window both pass with margin
        (norm 0.0163 of 0.05, std 0.0013 of 0.02).
    current_bias:
        (dim,) bias already in effect while ``Z`` was recorded. ``None`` means
        zeros, i.e. ``Z`` was decoded uncalibrated.

    Returns
    -------
    :class:`CalibrationResult`. Apply ``bias`` by assigning it to
    ``IMUDecoder.b`` and re-decoding, which is what the live path's
    ``apply_calibration`` does -- so offline and live share one mechanism
    rather than one subtracting after the fact.
    """
    t_a = np.asarray(t, dtype=np.float64).reshape(-1)
    Z_a = np.atleast_2d(np.asarray(Z, dtype=np.float64))
    if Z_a.shape[0] != t_a.size:
        raise ValueError(f"t has {t_a.size} samples, Z has {Z_a.shape[0]}")
    exp = np.asarray(expected_rest, dtype=np.float64).reshape(-1)
    dim = exp.size
    if Z_a.size and Z_a.shape[1] != dim:
        raise ValueError(f"Z has {Z_a.shape[1]} channels, expected_rest has {dim}")
    b0 = np.zeros(dim) if current_bias is None else np.asarray(current_bias, dtype=np.float64)
    g = np.asarray(gyro_idx, dtype=np.intp)

    rest = t_a < rest_s
    n_rest = int(rest.sum())
    gyro_norm = (
        float(np.linalg.norm(Z_a[rest][:, g], axis=1).max()) if n_rest and g.size else np.inf
    )
    if n_rest < min_samples or gyro_norm > max_gyro_norm:
        # Same diagnostic the notebook cell raised, so the message a reader has
        # seen before does not change -- only who decides it is fatal.
        return CalibrationResult(
            bias=b0.copy(), scale=np.ones(dim), R=np.zeros((dim, dim)), valid=False,
            note=(
                f"rest window [0, {rest_s}) s is not still: {n_rest} samples, "
                f"max gyro norm {gyro_norm:.3f} rad/s (limit {max_gyro_norm}). "
                f"Shorten rest_s."
            ),
        )
    return calibration_from_samples(
        Z_a[rest],
        b0,
        gyro_idx=g,
        acc_idx=acc_idx,
        expected_rest=exp,
        still_gyro_std=still_gyro_std,
        min_samples=min_samples,
    )
