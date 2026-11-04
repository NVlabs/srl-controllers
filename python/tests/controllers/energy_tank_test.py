# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Independent work-balance oracles for the public CPU/CUDA energy tank."""

# Standard Library
import ctypes
import math

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl
from nvidia.srl.controllers import equations as eq


def make_tank(runtime, dtype, batch=2, noise=0.0, upper=1.0):
    config = ctl.SignalBlockConfig(
        runtime.device, ctl.DType.FLOAT32 if dtype == np.float32 else ctl.DType.FLOAT64, batch, 1
    )
    parameters = ctl.EnergyTankParameters(
        *(
            ctl.shared(runtime.array(np.array(v, dtype=dtype)))
            for v in (upper, 0.01, upper, 0.1, noise)
        )
    )
    storage = ctl.EnergyTankStorage(
        runtime.zeros(batch, dtype),
        runtime.zeros(batch, dtype),
        runtime.zeros(batch, np.uint64),
        runtime.zeros(batch, np.uint64),
        ctl.EnergyTankState(runtime.zeros(batch, dtype)),
    )
    tank = ctl.EnergyTank(config, parameters, storage, runtime.context)
    tank.reset(context=runtime.context)
    return tank


def set_state(runtime, destination, value):
    if runtime.device == ctl.Device.CPU:
        np.copyto(destination, value)
    else:
        runtime.synchronize()
        value = np.ascontiguousarray(value)
        runtime.check(
            runtime.library.cudaMemcpy(
                ctypes.c_void_p(destination.pointer.value),
                ctypes.c_void_p(value.ctypes.data),
                value.nbytes,
                1,
            )
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "energy,power,extra,dt,noise",
    [
        (1, -1, 0, 0.1, 0),
        (0.11, -200, 0, 0.001, 0),
        (0.01, -1, 0, 0.1, 0),
        (0.01, 1, 0, 0.1, 0),
        (0.01, 0, 2, 0.1, 0),
        (0.11, -200, 200, 0.001, 0),
        (0.06, -1, 0, 0.001, 0),
        (0.06, 1, 0, 0.001, 0),
        (1, -0.5, 0, 1, 1),
        (0.5, 0.5, 0, 1, 1),
        (0.9, 100, 100, 1, 0),
    ],
)
def test_scale_and_energy_match_work_balance(runtime, dtype, energy, power, extra, dt, noise):
    tank = make_tank(runtime, dtype, noise=noise)
    set_state(runtime, tank.storage.state.state_variable, np.full(2, math.sqrt(2 * energy), dtype))
    stored_start = 0.5 * runtime.numpy(tank.storage.state.state_variable).astype(np.float64) ** 2
    inputs = [runtime.array(np.full(2, v, dtype)) for v in (power, extra, dt)]
    runtime.captured_step(lambda: tank.step(*inputs, context=runtime.context))
    scale = runtime.numpy(tank.scale).astype(np.float64)
    end = 0.5 * runtime.numpy(tank.storage.state.state_variable).astype(np.float64) ** 2
    start = np.clip(stored_start, 0.01, 1)
    alpha = np.where(start >= 0.11, 1, 0.5 * (1 - np.cos(np.pi * (start - 0.01) / 0.1)))
    filtered = 0 if abs(power) < noise else power
    nominal = alpha
    requested = max(-power, 0) * dt
    expected_scale = np.minimum(nominal, (start - 0.01) / requested) if requested else nominal
    spent = scale * requested
    credit = np.minimum((scale * max(filtered, 0) + extra) * dt, 1 - (start - spent))
    tolerance = 32 * np.finfo(dtype).eps
    np.testing.assert_allclose(scale, expected_scale, atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(end, start - spent + credit, atol=tolerance, rtol=0)
    assert np.all(spent <= start - 0.01 + tolerance)
    assert np.all(end >= 0.01 - tolerance) and np.all(end <= 1 + tolerance)
    assert not np.any(runtime.numpy(tank.status) & ctl.STATUS_SAFE_MODE)
    np.testing.assert_allclose(runtime.numpy(tank.filtered_power), filtered)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_repeated_spending_never_creates_energy(runtime, dtype):
    tank = make_tank(runtime, dtype)
    balance = 0.5 * runtime.numpy(tank.storage.state.state_variable).astype(np.float64) ** 2
    power = runtime.array(np.array([-20, 0.3], dtype))
    extra = runtime.array(np.zeros(2, dtype))
    dt = runtime.array(np.full(2, 0.01, dtype))
    for _ in range(100):
        runtime.captured_step(lambda: tank.step(power, extra, dt, context=runtime.context))
        scale = runtime.numpy(tank.scale).astype(np.float64)
        previous = balance.copy()
        spent = scale * np.array([0.2, 0])
        credit = np.minimum(scale * np.array([0, 0.003]), 1 - (previous - spent))
        balance += credit - spent
    energy = 0.5 * runtime.numpy(tank.storage.state.state_variable).astype(np.float64) ** 2
    np.testing.assert_allclose(energy, balance, atol=32 * np.finfo(dtype).eps * 100, rtol=0)
    assert np.all(energy >= 0.01 - 32 * np.finfo(dtype).eps)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "kind",
    [
        "negative_recharge",
        "nonfinite",
        "zero_dt",
        "negative_dt",
        "low_state",
        "high_state",
        "negative_state",
        "overflow",
    ],
)
def test_invalid_row_holds_state_and_other_row_remains_operational(runtime, dtype, kind):
    tank = make_tank(runtime, dtype)
    power, extra, dt = np.array([-1, -1], dtype), np.zeros(2, dtype), np.full(2, 0.01, dtype)
    state = runtime.numpy(tank.storage.state.state_variable).copy()
    if kind == "negative_recharge":
        extra[0] = -1
    if kind == "nonfinite":
        power[0] = np.nan
    if kind == "zero_dt":
        dt[0] = 0
    if kind == "negative_dt":
        dt[0] = -1
    if kind == "low_state":
        state[0] = 0.01
    if kind == "high_state":
        state[0] = 2
    if kind == "negative_state":
        state[0] = -1
    if kind == "overflow":
        power[0], dt[0] = -np.finfo(dtype).max, 2
    set_state(runtime, tank.storage.state.state_variable, state)
    inputs = [runtime.array(v) for v in (power, extra, dt)]
    runtime.captured_step(lambda: tank.step(*inputs, context=runtime.context))
    status = runtime.numpy(tank.status)
    assert int(status[0]) & ctl.STATUS_SAFE_MODE and not int(status[1]) & ctl.STATUS_SAFE_MODE
    assert runtime.numpy(tank.scale)[0] == 0
    assert runtime.numpy(tank.storage.state.state_variable)[0] == state[0]
    assert runtime.numpy(tank.storage.state.state_variable)[1] < state[1]
    tank.reset(runtime.array(np.array([True, False])), runtime.context)
    after = runtime.numpy(tank.storage.state.state_variable)
    assert after[0] == pytest.approx(math.sqrt(2), rel=4 * np.finfo(dtype).eps)
    assert after[1] < state[1]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_checked_equation_rejects_invalid_domain_and_limits_overdraw(runtime, dtype):
    config = eq.EquationConfig(
        runtime.device, ctl.DType.FLOAT32 if dtype == np.float32 else ctl.DType.FLOAT64, 2
    )
    outputs = eq.EnergyTankStepOutputs(*(runtime.zeros(2, t) for t in (dtype, dtype, dtype, bool)))
    status = runtime.zeros(2, np.uint64)
    inputs = dict(
        zip(
            (
                "state_variable",
                "power_signal",
                "additional_power",
                "t_delta",
                "upper_energy_limit",
                "lower_energy_limit",
                "empty_threshold",
                "power_noise",
            ),
            (
                np.array([math.sqrt(0.22), 2], dtype),
                np.full(2, -200, dtype),
                np.zeros(2, dtype),
                np.full(2, 0.001, dtype),
                1,
                0.01,
                0.1,
                0,
            ),
            strict=True,
        )
    )
    inputs = {
        k: (
            ctl.per_batch(runtime.array(v))
            if isinstance(v, np.ndarray)
            else ctl.shared(runtime.array(np.array(v, dtype)))
        )
        for k, v in inputs.items()
    }
    runtime.captured_step(
        lambda: eq.energy_tank_step(
            config, **inputs, outputs=outputs, status=status, context=runtime.context
        )
    )
    assert runtime.numpy(outputs.scale)[0] == pytest.approx(0.5, abs=32 * np.finfo(dtype).eps)
    assert int(runtime.numpy(status)[1]) & ctl.STATUS_SAFE_MODE
    assert runtime.numpy(outputs.state_variable)[1] == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_large_finite_initial_energy_does_not_overflow_reset(runtime, dtype):
    upper = float(np.finfo(dtype).max) * 0.75
    tank = make_tank(runtime, dtype, upper=upper)
    power, extra, dt = [runtime.array(np.full(2, v, dtype)) for v in (10, 0, 0.1)]
    runtime.captured_step(lambda: tank.step(power, extra, dt, context=runtime.context))
    state = runtime.numpy(tank.storage.state.state_variable).astype(np.float64)
    assert np.all(np.isfinite(state))
    np.testing.assert_allclose(0.5 * state * state, upper, rtol=32 * np.finfo(dtype).eps)
    assert not np.any(runtime.numpy(tank.status) & ctl.STATUS_SAFE_MODE)
