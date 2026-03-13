// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/argsort_kernel.h"

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/gpu/gpu_info.h"
#include "paddle/phi/backends/gpu/gpu_launch_config.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/platform/cuda_graph_with_memory_pool.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/cub.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/transpose_kernel.h"

#ifdef __HIPCC__
namespace rocprim {
namespace detail {
template <>
struct radix_key_codec_base<phi::float16>
    : radix_key_codec_integral<phi::float16, uint16_t> {};

template <>
struct radix_key_codec_base<phi::bfloat16>
    : radix_key_codec_integral<phi::bfloat16, uint16_t> {};

#if HIP_VERSION >= 50400000
template <>
struct float_bit_mask<phi::float16> : float_bit_mask<rocprim::half> {};

template <>
struct float_bit_mask<phi::bfloat16> : float_bit_mask<rocprim::bfloat16> {};
#endif
}  // namespace detail
}  // namespace rocprim
#else
// set cub base traits in order to handle float16
namespace cub {
template <>
struct NumericTraits<phi::float16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::float16> {};

template <>
struct NumericTraits<phi::bfloat16>
    : BaseTraits<FLOATING_POINT, true, false, uint16_t, phi::bfloat16> {};
}  // namespace cub

#endif

namespace phi {

// Iter for move to next row
struct SegmentOffsetIter {
  EIGEN_DEVICE_FUNC
  explicit SegmentOffsetIter(int num_cols) : num_cols_(num_cols) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  }

  int num_cols_;
};

template <typename T, typename IndType>
__global__ void merge_kernel(const T* A,
                             size_t sizeA,
                             const T* B,
                             size_t sizeB,
                             const IndType* ids_A,
                             const IndType* ids_B,
                             T* out,
                             IndType* out_ids,
                             bool descending) {
  int64_t thread =
      static_cast<int64_t>(blockDim.x) * static_cast<int64_t>(gridDim.x);
  int64_t num_per_thread = (sizeA + sizeB + thread) / thread;
  for (int64_t offset = 0; offset < num_per_thread; offset++) {
    size_t idx =
        static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) +
        static_cast<size_t>(threadIdx.x) + offset * thread;
    size_t total = sizeA + sizeB;
    if (idx >= total) return;
    size_t left = (idx > sizeB) ? idx - sizeB : 0;
    size_t right = (idx < sizeA) ? idx : sizeA;
    while (left < right) {
      size_t mid = (left + right) / 2;
      size_t b_idx = idx - mid;

      T A_mid, B_bidx;
      if (descending) {
        A_mid = (mid >= sizeA) ? std::numeric_limits<T>::lowest() : A[mid];
        B_bidx = (b_idx >= sizeB) ? std::numeric_limits<T>::lowest() : B[b_idx];
      } else {
        A_mid = (mid >= sizeA) ? std::numeric_limits<T>::max() : A[mid];
        B_bidx = (b_idx >= sizeB) ? std::numeric_limits<T>::max() : B[b_idx];
      }

      if (descending ? (A_mid >= B_bidx) : (A_mid <= B_bidx))
        left = mid + 1;
      else
        right = mid;
    }

    size_t a_idx = left;
    size_t b_idx = idx - a_idx;
    if (a_idx >= sizeA) {
      if (descending ? (A[sizeA - 1] < B[b_idx]) : (A[sizeA - 1] > B[b_idx])) {
        out[idx] = A[sizeA - 1];
        out_ids[idx] = ids_A[sizeA - 1];
      } else {
        out[idx] = B[b_idx];
        out_ids[idx] = ids_B[b_idx];
      }
    } else if (b_idx >= sizeB) {
      out[idx] = A[a_idx];
      out_ids[idx] = ids_A[a_idx];
    } else {
      if (descending ? (A[a_idx] >= B[b_idx]) : (A[a_idx] <= B[b_idx])) {
        out[idx] = A[a_idx];
        out_ids[idx] = ids_A[a_idx];
      } else if (descending ? (a_idx > 0 && (A[a_idx - 1] < B[b_idx]))
                            : (a_idx > 0 && (A[a_idx - 1] > B[b_idx]))) {
        out[idx] = A[a_idx - 1];
        out_ids[idx] = ids_A[a_idx - 1];
      } else {
        out[idx] = B[b_idx];
        out_ids[idx] = ids_B[b_idx];
      }
    }
  }
}

template <typename T>
static __global__ void FillIndex(T* indices, T num_rows, T num_cols) {
  int col_id = threadIdx.x;
  int row_id = blockIdx.x;

  for (T j = row_id; j < num_rows; j += gridDim.x) {
    for (T i = col_id; i < num_cols; i += blockDim.x) {
      indices[j * num_cols + i] = i;
    }
  }
}

#define CUB_ARGSORT_WRAPPER(func, ...)                                        \
  {                                                                           \
    size_t temp_storage_bytes = 0;                                            \
    {                                                                         \
      /* The nullptr-query call must not be captured in a CUDA graph since it \
       * does not enqueue any GPU work but merely returns a size. Use         \
       * SkipCUDAGraphCaptureGuard to temporarily pause capture. */           \
      paddle::platform::SkipCUDAGraphCaptureGuard skip_guard;                 \
      PADDLE_ENFORCE_GPU_SUCCESS(                                             \
          func(nullptr, temp_storage_bytes, __VA_ARGS__));                    \
    }                                                                         \
    int64_t temp_size = static_cast<int64_t>(temp_storage_bytes);             \
    PADDLE_ENFORCE_GT(                                                        \
        temp_size,                                                            \
        0,                                                                    \
        common::errors::InvalidArgument(                                      \
            "Argsort temp storage size is %d, but should be greater than 0.", \
            temp_size));                                                      \
    /* Allocate temp storage via phi::memory_utils::Alloc whose AllocationPtr \
     * destructor calls FreeImpl through the allocator. Since cudaFree is not \
     * allowed while a stream is capturing, we must free the allocation while \
     * capture is temporarily paused (inside another guard). */               \
    phi::Allocator::AllocationPtr temp_alloc =                                \
        phi::memory_utils::Alloc(dev_ctx.GetPlace(), temp_size);              \
    void* temp_ptr = temp_alloc->ptr();                                       \
    PADDLE_ENFORCE_GPU_SUCCESS(                                               \
        func(temp_ptr, temp_storage_bytes, __VA_ARGS__));                     \
    /* Release temp_alloc while capture is paused to avoid cudaFree being     \
     * called on a capturing stream. */                                       \
    {                                                                         \
      paddle::platform::SkipCUDAGraphCaptureGuard skip_guard2;                \
      temp_alloc.reset();                                                     \
    }                                                                         \
  }

#define PREDICATE_CUB_ARGSORT(predicate, if_func, else_func, ...) \
  if (predicate)                                                  \
    CUB_ARGSORT_WRAPPER(if_func, __VA_ARGS__)                     \
  else                                                            \
    CUB_ARGSORT_WRAPPER(else_func, __VA_ARGS__)

// Sort by flag descending, True: descending. False: Ascending.
// Default is false.
template <typename T, typename IndType>
void ArgFullSort(const GPUContext& dev_ctx,
                 const DenseTensor* input,
                 DenseTensor* output,
                 DenseTensor* indices,
                 const int64_t num_rows,
                 const int64_t num_cols,
                 const bool descending) {
  PADDLE_ENFORCE_LE_INT_MAX(num_cols, "num_cols");

  auto cu_stream = dev_ctx.stream();
  auto ComputeBlockSize = [](IndType col) {
    if (col > 512)
      return 1024;
    else if (col > 256 && col <= 512)
      return 512;
    else if (col > 128 && col <= 256)
      return 256;
    else
      return 128;
  };
  const int block_size = ComputeBlockSize(num_cols);
  const int64_t maxGridDimX = dev_ctx.GetCUDAMaxGridDimSize()[0];

  const T* inp = input->data<T>();
  IndType* sorted_indices_ptr = indices->data<IndType>();

  // create iter for counting input
  cub::CountingInputIterator<IndType> counting_iter(0);
  // segment_offset is used for move to next row
  cub::TransformInputIterator<IndType,
                              SegmentOffsetIter,
                              cub::CountingInputIterator<IndType>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));

  // num_rows is the total segments to be sorted
  constexpr int64_t max_elements = 1 << 30;
  const int64_t total_elements = num_cols * num_rows;
  const int64_t segment_size = num_cols;
  const int64_t element_per_call = std::min(max_elements, total_elements);

  // make sure element_per_call >= segment_size
  const int64_t adjusted_elements_per_call =
      std::max(max_elements, segment_size);

  // make sure batch size is the multiple of segment_size
  const int64_t batch_size =
      (adjusted_elements_per_call / segment_size) * segment_size;
  int64_t offset = 0;
  T* sorted_out_ptr = output->data<T>();
  phi::Allocator::AllocationPtr input_indices_alloc;
  IndType* ind_ptr = nullptr;

  while (offset < total_elements) {
    const int64_t n_elements = std::min(batch_size, total_elements - offset);
    const int64_t n_segments = n_elements / segment_size;

    // allocate a temporary storage for input indices, with shape:
    // [num_segments = n_elements / segment_size, segment_size]
    // Use AllocationPtr (instead of DenseTensor) so we can free it inside a
    // SkipCUDAGraphCaptureGuard, ensuring cudaFree is never called on a
    // capturing stream.
    if (input_indices_alloc == nullptr) {
      input_indices_alloc = phi::memory_utils::Alloc(
          dev_ctx.GetPlace(), n_segments * segment_size * sizeof(IndType));
      ind_ptr = reinterpret_cast<IndType*>(input_indices_alloc->ptr());
    }
    const int64_t grid_size = std::min(n_segments, maxGridDimX);
    // Init a index array
    FillIndex<<<grid_size, block_size, 0, cu_stream>>>(
        ind_ptr, n_segments, segment_size);

    PREDICATE_CUB_ARGSORT(descending,
                          cub::DeviceSegmentedRadixSort::SortPairsDescending,
                          cub::DeviceSegmentedRadixSort::SortPairs,
                          inp + offset,
                          sorted_out_ptr + offset,
                          ind_ptr,
                          sorted_indices_ptr + offset,
                          n_elements,
                          n_segments,
                          segment_offsets_t,
                          segment_offsets_t + 1,
                          0,
                          sizeof(T) * 8,
                          cu_stream);
    offset += n_elements;
  }
  // Free temporary index buffer while capture is paused so cudaFree is not
  // called on a capturing stream.
  if (input_indices_alloc != nullptr) {
    paddle::platform::SkipCUDAGraphCaptureGuard skip_free_guard;
    input_indices_alloc.reset();
  }
}
// Sort a contiguous range [start, end) of keys/values using CUB
// DeviceRadixSort, which is a stable sort and is compatible with CUDA graph
// capture (the nullptr-query phase is guarded by SkipCUDAGraphCaptureGuard
// inside CUB_ARGSORT_WRAPPER).
template <typename T, typename IndType>
void PerSort(const GPUContext& dev_ctx,
             T* out_data,
             IndType* ids_data,
             IndType start,
             IndType end,
             bool stable,
             bool descending) {
  const IndType count = end - start;
  if (count <= 0) return;

  // CUB DeviceRadixSort requires separate input/output buffers.
  // Allocate via phi::memory_utils::Alloc and manually free inside a
  // SkipCUDAGraphCaptureGuard so that cudaFree is never called on a
  // capturing stream.
  phi::Allocator::AllocationPtr keys_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), count * sizeof(T));
  phi::Allocator::AllocationPtr vals_alloc =
      phi::memory_utils::Alloc(dev_ctx.GetPlace(), count * sizeof(IndType));
  T* tmp_keys = reinterpret_cast<T*>(keys_alloc->ptr());
  IndType* tmp_vals = reinterpret_cast<IndType*>(vals_alloc->ptr());

  if (descending) {
    CUB_ARGSORT_WRAPPER(cub::DeviceRadixSort::SortPairsDescending,
                        out_data + start,
                        tmp_keys,
                        ids_data + start,
                        tmp_vals,
                        static_cast<int>(count),
                        0,
                        sizeof(T) * 8,
                        dev_ctx.stream());
  } else {
    CUB_ARGSORT_WRAPPER(cub::DeviceRadixSort::SortPairs,
                        out_data + start,
                        tmp_keys,
                        ids_data + start,
                        tmp_vals,
                        static_cast<int>(count),
                        0,
                        sizeof(T) * 8,
                        dev_ctx.stream());
  }
  // Copy sorted results back in-place using cudaMemcpyAsync, which is
  // compatible with CUDA graph capture.
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(out_data + start,
                                             tmp_keys,
                                             count * sizeof(T),
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(ids_data + start,
                                             tmp_vals,
                                             count * sizeof(IndType),
                                             cudaMemcpyDeviceToDevice,
                                             dev_ctx.stream()));
  // Free temp buffers while capture is paused so cudaFree is not called on
  // a capturing stream.
  {
    paddle::platform::SkipCUDAGraphCaptureGuard skip_free_guard;
    keys_alloc.reset();
    vals_alloc.reset();
  }
}

template <typename T, typename Context>
void ArgsortKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   int axis,
                   bool descending,
                   bool stable,
                   DenseTensor* output,
                   DenseTensor* indices) {
  auto in_dims = input.dims();
  auto rank = in_dims.size();

  if (input.numel() == 0) {
    output->Resize(in_dims);
    indices->Resize(in_dims);
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    return;
  }

  axis = (axis < 0) ? (in_dims.size() + axis) : axis;
  const T* in_data = input.data<T>();
  auto size = input.numel();

  if (rank == 0) {
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    Copy<Context>(dev_ctx, input, dev_ctx.GetPlace(), false, output);
    funcs::set_constant(dev_ctx, indices, static_cast<int64_t>(0));
    return;
  }

  // Use CUB for parallel acceleration when the input size is equal to the
  // length of the 'axis' dimension.
  // Skip this path during CUDA graph capture because cub::DeviceRadixSort
  // internally uses thrust::parallel_for which calls cudaStreamSynchronize,
  // an operation that is not permitted during stream capture. Fall through to
  // the full-sort (DeviceSegmentedRadixSort) path instead, which is
  // CUDA-graph-compatible.
  if (size == in_dims[axis] &&
      !phi::backends::gpu::CUDAGraph::IsThisThreadCapturing()) {
    T* out_data = dev_ctx.template Alloc<T>(output);
    int64_t* ids_data = dev_ctx.template Alloc<int64_t>(indices);
    auto cu_stream = dev_ctx.stream();

    // Use a custom kernel to fill sequence 0..size-1 into ids_data.
    // thrust::sequence uses internal synchronization that is incompatible
    // with CUDA graph capture.
    {
      auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, size);
      FillIndex<int64_t><<<config.block_per_grid.x,
                           config.thread_per_block.x,
                           0,
                           cu_stream>>>(ids_data, 1, size);
    }
    // Use cudaMemcpyAsync instead of thrust::copy; memcpy is always
    // capturable in CUDA graphs.
    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(out_data,
                                               in_data,
                                               size * sizeof(T),
                                               cudaMemcpyDeviceToDevice,
                                               cu_stream));
    const int64_t per_number = (1LL << 31) - 1;
    int64_t start = 0;
    int64_t end = std::min(start + per_number, size);
    if (end == size) {
      PerSort<T, int64_t>(
          dev_ctx, out_data, ids_data, start, end, stable, descending);
    } else {
      // Sorting the segments and then merging them
      DenseTensor temp;
      DenseTensor ids;
      temp.Resize(in_dims);
      ids.Resize(in_dims);
      T* temp_data = dev_ctx.template Alloc<T>(&temp);
      int64_t* temp_ids = dev_ctx.template Alloc<int64_t>(&ids);

      while (start != size) {
        PerSort<T, int64_t>(
            dev_ctx, out_data, ids_data, start, end, stable, descending);
        if (start != 0) {
          auto config = phi::backends::gpu::GetGpuLaunchConfig1D(dev_ctx, end);
          merge_kernel<<<config.block_per_grid.x,
                         config.thread_per_block.x,
                         0,
                         cu_stream>>>(out_data,
                                      start,
                                      out_data + start,
                                      end - start,
                                      ids_data,
                                      ids_data + start,
                                      temp_data,
                                      temp_ids,
                                      descending);
          PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(ids_data,
                                                     temp_ids,
                                                     end * sizeof(int64_t),
                                                     cudaMemcpyDeviceToDevice,
                                                     cu_stream));
          PADDLE_ENFORCE_GPU_SUCCESS(cudaMemcpyAsync(out_data,
                                                     temp_data,
                                                     end * sizeof(T),
                                                     cudaMemcpyDeviceToDevice,
                                                     cu_stream));
        }
        start = end;
        end = std::min(start + per_number, size);
      }
    }
    return;
  }

  // Special case for full sort, speedup ~190x.
  if (axis == -1 || axis + 1 == in_dims.size()) {
    const int64_t input_height =
        common::product(slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t input_width = in_dims[in_dims.size() - 1];
    dev_ctx.template Alloc<int64_t>(indices);
    dev_ctx.template Alloc<T>(output);
    ArgFullSort<T, int64_t>(dev_ctx,
                            &input,
                            output,
                            indices,
                            input_height,
                            input_width,
                            descending);
  } else {
    // if not full sort, do transpose first
    std::vector<int> trans;
    for (int i = 0; i < axis; i++) {
      trans.push_back(i);
    }
    trans.push_back(in_dims.size() - 1);
    for (int i = axis + 1; i < in_dims.size() - 1; i++) {
      trans.push_back(i);
    }
    trans.push_back(axis);
    DDim trans_dims(in_dims);
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
    }

    DenseTensor trans_inp;
    trans_inp.Resize(trans_dims);
    T* trans_inp_data = dev_ctx.template Alloc<T>(&trans_inp);
    // Do transpose
    TransposeKernel<T, Context>(dev_ctx, input, trans, &trans_inp);

    const int64_t input_height =
        common::product(slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];

    DenseTensor tmp_out;
    tmp_out.Resize(trans_dims);
    dev_ctx.template Alloc<T>(&tmp_out);

    DenseTensor tmp_indices;
    // temp indices for sorting
    tmp_indices.Resize(trans_dims);
    dev_ctx.template Alloc<int64_t>(&tmp_indices);

    ArgFullSort<T, int64_t>(dev_ctx,
                            &trans_inp,
                            &tmp_out,
                            &tmp_indices,
                            input_height,
                            input_width,
                            descending);
    // delay output allocation until after transpose, to avoid
    // allocating too much memory
    dev_ctx.template Alloc<T>(output);
    dev_ctx.template Alloc<int64_t>(indices);
    // transpose back
    TransposeKernel<T, Context>(dev_ctx, tmp_out, trans, output);
    TransposeKernel<int64_t, Context>(dev_ctx, tmp_indices, trans, indices);
    return;
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(argsort,
                   GPU,
                   ALL_LAYOUT,
                   phi::ArgsortKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   uint8_t,
                   int16_t,
                   phi::float16,
                   phi::bfloat16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::INT64);
}
