# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that argsort and cumsum can be captured and replayed in CUDA graphs."""

import unittest

import numpy as np

import paddle
from paddle.device.cuda.graphs import CUDAGraph


def can_use_cuda_graph():
    return (
        paddle.is_compiled_with_cuda()
        and not paddle.is_compiled_with_rocm()
        and float(paddle.version.cuda()) >= 11.0
    )


def _cumsum(x, axis, flatten=False, exclusive=False, reverse=False):
    """Call the C++ cumsum kernel directly to support exclusive and reverse."""
    # When flatten=True, axis is ignored by the kernel; pass 0 as placeholder.
    if axis is None:
        axis = 0
    return paddle._C_ops.cumsum(x, axis, flatten, exclusive, reverse)


@unittest.skipIf(
    not can_use_cuda_graph(),
    "only support CUDA >= 11.0 and non-ROCm builds",
)
class TestCUDAGraphArgsort(unittest.TestCase):
    def setUp(self):
        paddle.set_flags(
            {
                'FLAGS_allocator_strategy': 'auto_growth',
                'FLAGS_sync_nccl_allreduce': False,
                'FLAGS_cudnn_deterministic': True,
                'FLAGS_use_stream_safe_cuda_allocator': False,
            }
        )

    def _run_argsort_cuda_graph(self, x_tensor, axis, descending, stable):
        """Capture argsort in a CUDA graph and replay it, returning indices."""
        g = CUDAGraph()
        g.capture_begin()
        indices = paddle.argsort(
            x_tensor, axis=axis, descending=descending, stable=stable
        )
        g.capture_end()

        # First replay
        g.replay()
        idx_np = indices.numpy().copy()
        g.reset()
        return idx_np

    def test_argsort_last_axis_ascending(self):
        """Full-sort path (last axis), ascending."""
        shape = [4, 8]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)
        idx_np = self._run_argsort_cuda_graph(
            x, axis=-1, descending=False, stable=False
        )
        expected_idx = np.argsort(np_data, axis=-1, kind='quicksort')
        np.testing.assert_array_equal(idx_np, expected_idx)

    def test_argsort_last_axis_descending(self):
        """Full-sort path (last axis), descending."""
        shape = [4, 8]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)
        idx_np = self._run_argsort_cuda_graph(
            x, axis=-1, descending=True, stable=False
        )
        expected_idx = np.argsort(np_data, axis=-1, kind='quicksort')[:, ::-1]
        np.testing.assert_array_equal(idx_np, expected_idx)

    def test_argsort_non_last_axis(self):
        """Transpose path (non-last axis)."""
        shape = [3, 5, 4]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)
        idx_np = self._run_argsort_cuda_graph(
            x, axis=1, descending=False, stable=True
        )
        expected_idx = np.argsort(np_data, axis=1, kind='stable')
        np.testing.assert_array_equal(idx_np, expected_idx)

    def test_argsort_1d_thrust_path(self):
        """1D tensor: size == in_dims[axis], full-sort path during capture."""
        shape = [64]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)
        idx_np = self._run_argsort_cuda_graph(
            x, axis=0, descending=False, stable=True
        )
        expected_idx = np.argsort(np_data, axis=0, kind='stable')
        np.testing.assert_array_equal(idx_np, expected_idx)

    def test_argsort_replay_multiple_times(self):
        """Verify that replaying with updated input data produces correct results."""
        shape = [4, 8]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)

        g = CUDAGraph()
        g.capture_begin()
        indices = paddle.argsort(x, axis=-1, descending=False)
        g.capture_end()

        for _ in range(5):
            new_np = np.random.rand(*shape).astype("float32")
            x.copy_(paddle.to_tensor(new_np), False)
            g.replay()
            idx_np = indices.numpy()
            expected_idx = np.argsort(new_np, axis=-1)
            np.testing.assert_array_equal(idx_np, expected_idx)

        g.reset()

    def test_argsort_int_dtype(self):
        """Test with integer dtype."""
        shape = [3, 6]
        np_data = np.random.randint(0, 100, shape).astype("int32")
        x = paddle.to_tensor(np_data)
        idx_np = self._run_argsort_cuda_graph(
            x, axis=-1, descending=False, stable=True
        )
        expected_idx = np.argsort(np_data, axis=-1, kind='stable')
        np.testing.assert_array_equal(idx_np, expected_idx)


@unittest.skipIf(
    not can_use_cuda_graph(),
    "only support CUDA >= 11.0 and non-ROCm builds",
)
class TestCUDAGraphCumsum(unittest.TestCase):
    def setUp(self):
        paddle.set_flags(
            {
                'FLAGS_allocator_strategy': 'auto_growth',
                'FLAGS_sync_nccl_allreduce': False,
                'FLAGS_cudnn_deterministic': True,
                'FLAGS_use_stream_safe_cuda_allocator': False,
            }
        )

    def _run_cumsum_cuda_graph(
        self, x_tensor, axis, flatten=False, exclusive=False, reverse=False
    ):
        g = CUDAGraph()
        g.capture_begin()
        out = _cumsum(
            x_tensor,
            axis=axis,
            flatten=flatten,
            exclusive=exclusive,
            reverse=reverse,
        )
        g.capture_end()
        g.replay()
        out_np = out.numpy().copy()
        g.reset()
        return out_np

    def test_cumsum_1d(self):
        """1D tensor cumsum (ThrustCumsumKernel path, T==MT)."""
        np_data = np.random.rand(32).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=0)
        np.testing.assert_allclose(
            out_np, np.cumsum(np_data, axis=0), rtol=1e-5
        )

    def test_cumsum_2d_axis0(self):
        """2D tensor, non-last-axis path (ScanKernel with transpose)."""
        np_data = np.random.rand(4, 8).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=0)
        np.testing.assert_allclose(
            out_np, np.cumsum(np_data, axis=0), rtol=1e-5
        )

    def test_cumsum_2d_last_axis(self):
        """2D tensor, last-axis path."""
        np_data = np.random.rand(4, 8).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=-1)
        np.testing.assert_allclose(
            out_np, np.cumsum(np_data, axis=-1), rtol=1e-5
        )

    def test_cumsum_reverse(self):
        """Reverse cumsum."""
        np_data = np.random.rand(4, 8).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=-1, reverse=True)
        expected = np.cumsum(np_data[:, ::-1], axis=-1)[:, ::-1]
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)

    def test_cumsum_exclusive(self):
        """Exclusive cumsum."""
        np_data = np.random.rand(4, 8).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=-1, exclusive=True)
        # exclusive cumsum: shift right by one, first element is 0
        expected = np.concatenate(
            [
                np.zeros((4, 1), dtype=np.float32),
                np.cumsum(np_data, axis=-1)[:, :-1],
            ],
            axis=-1,
        )
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)

    def test_cumsum_float16_type_promotion(self):
        """float16 -> float32 type promotion path (was using thrust::device_vector)."""
        np_data = np.random.rand(32).astype("float16")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=0)
        expected = np.cumsum(np_data.astype("float32"), axis=0).astype(
            "float16"
        )
        np.testing.assert_allclose(out_np, expected, rtol=1e-3)

    def test_cumsum_replay_multiple_times(self):
        """Verify replay with updated data."""
        shape = [4, 8]
        np_data = np.random.rand(*shape).astype("float32")
        x = paddle.to_tensor(np_data)

        g = CUDAGraph()
        g.capture_begin()
        out = _cumsum(x, axis=-1)
        g.capture_end()

        for _ in range(5):
            new_np = np.random.rand(*shape).astype("float32")
            x.copy_(paddle.to_tensor(new_np), False)
            g.replay()
            out_np = out.numpy()
            np.testing.assert_allclose(
                out_np, np.cumsum(new_np, axis=-1), rtol=1e-5
            )

        g.reset()

    def test_cumsum_flatten(self):
        """Cumsum with flatten=True."""
        np_data = np.random.rand(3, 4).astype("float32")
        x = paddle.to_tensor(np_data)
        out_np = self._run_cumsum_cuda_graph(x, axis=None, flatten=True)
        expected = np.cumsum(np_data.flatten())
        np.testing.assert_allclose(out_np, expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
