# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-2022-03-22
"""Array-provider-neutral CUDA fixtures; transfers happen only in the test harness."""

# Standard Library
import ctypes
import ctypes.util
import dataclasses
import os

# Third Party
import numpy as np
import pytest

# NVIDIA
import nvidia.srl.controllers as ctl


class CudaArray:
    """Minimal CUDA array interface owner for testing device-only public calls."""

    def __init__(self, runtime, value):
        """Allocate storage and explicitly upload the fixture values."""
        self.runtime = runtime
        self.host = np.array(value, copy=True, order="C")
        self.pointer = ctypes.c_void_p()
        runtime.check(runtime.library.cudaMalloc(ctypes.byref(self.pointer), self.host.nbytes))
        runtime.check(
            runtime.library.cudaMemcpy(
                self.pointer, ctypes.c_void_p(self.host.ctypes.data), self.host.nbytes, 1
            )
        )
        self.__cuda_array_interface__ = {
            "shape": self.host.shape,
            "typestr": self.host.dtype.str,
            "data": (self.pointer.value, False),
            "strides": None,
            "version": 3,
        }
        runtime.allocations.append(self)

    def numpy(self):
        """Explicitly synchronize and read device results for assertions."""
        self.runtime.synchronize()
        self.runtime.check(
            self.runtime.library.cudaMemcpy(
                ctypes.c_void_p(self.host.ctypes.data), self.pointer, self.host.nbytes, 2
            )
        )
        return self.host.copy()


class ArrayRuntime:
    """Test-owned stream, allocations and explicit transfers."""

    def __init__(self, device):
        """Create a test-owned nonblocking CUDA stream when requested."""
        self.device = device
        self.allocations = []
        self.context = None
        if device == ctl.Device.CUDA:
            library = ctypes.util.find_library("cudart")
            if library is None:
                for candidate in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
                    try:
                        self.library = ctypes.CDLL(candidate)
                        break
                    except OSError:
                        continue
                else:
                    raise RuntimeError("CONTROLLERS_TEST_CUDA requires the CUDA runtime")
            else:
                self.library = ctypes.CDLL(library)
            self.library.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
            self.library.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self.library.cudaFree.argtypes = [ctypes.c_void_p]
            self.library.cudaStreamCreateWithFlags.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_uint,
            ]
            self.library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
            self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
            self.stream = ctypes.c_void_p()
            self.check(self.library.cudaStreamCreateWithFlags(ctypes.byref(self.stream), 1))
            self.context = ctl.ExecutionContext(cuda_stream=self.stream.value)

    @staticmethod
    def check(code):
        """Fail qualification instead of silently skipping a missing/broken GPU."""
        assert code == 0, f"CUDA runtime error {code}"

    def array(self, value):
        """Copy test input to the configured device."""
        return (
            CudaArray(self, value) if self.device == ctl.Device.CUDA else np.array(value, copy=True)
        )

    def zeros(self, shape, dtype):
        """Allocate initialized caller-owned storage."""
        return self.array(np.zeros(shape, dtype=dtype))

    def numpy(self, value):
        """Read results explicitly."""
        return value.numpy() if isinstance(value, CudaArray) else value

    def tree(self, value):
        """Convert arrays in a typed parameter or storage tree."""
        if isinstance(value, np.ndarray):
            return self.array(value)
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(
                value,
                **{
                    field.name: self.tree(getattr(value, field.name))
                    for field in dataclasses.fields(value)
                },
            )
        return value

    def captured_step(self, step):
        """Capture and replay a public step on the caller stream."""
        if self.device != ctl.Device.CUDA:
            step()
            return
        graph, executable = ctypes.c_void_p(), ctypes.c_void_p()
        self.library.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.cudaStreamEndCapture.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.library.cudaGraphInstantiateWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_ulonglong,
        ]
        self.library.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.library.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.check(self.library.cudaStreamBeginCapture(self.stream, 0))
        step()
        self.check(self.library.cudaStreamEndCapture(self.stream, ctypes.byref(graph)))
        try:
            self.check(
                self.library.cudaGraphInstantiateWithFlags(ctypes.byref(executable), graph, 0)
            )
            self.check(self.library.cudaGraphLaunch(executable, self.stream))
            self.synchronize()
        finally:
            if executable.value:
                self.check(self.library.cudaGraphExecDestroy(executable))
            self.check(self.library.cudaGraphDestroy(graph))

    def synchronize(self):
        """Wait only at test observation points."""
        if self.device == ctl.Device.CUDA:
            self.check(self.library.cudaStreamSynchronize(self.stream))

    def close(self):
        """Release test-owned resources after work completes."""
        self.synchronize()
        for array in self.allocations:
            self.check(self.library.cudaFree(array.pointer))
        self.allocations.clear()
        if self.device == ctl.Device.CUDA:
            self.check(self.library.cudaStreamDestroy(self.stream))


@pytest.fixture(
    params=[ctl.Device.CPU]
    + ([ctl.Device.CUDA] if os.environ.get("CONTROLLERS_TEST_CUDA") == "1" else [])
)
def runtime(request):
    """Enable mandatory real CUDA execution with CONTROLLERS_TEST_CUDA=1."""
    instance = ArrayRuntime(request.param)
    try:
        yield instance
    finally:
        instance.close()
