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

#pragma once

#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
inline void ResizeToChannelFirst(const DeviceContext& context,
                                 const Tensor* input,
                                 Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[4];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    in_dims_vec[4] = input->dims()[3];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[3];
    in_dims_vec[2] = input->dims()[1];
    in_dims_vec[3] = input->dims()[2];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  } else if (dim == 1) {
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  }
}

template <typename DeviceContext, typename T>
inline void ResizeToChannelLast(const DeviceContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[4];
    in_dims_vec[4] = input->dims()[1];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);

  } else if (dim == 2) {
    // input
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[3];
    in_dims_vec[3] = input->dims()[1];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  } else if (dim == 1) {
    transformed_input->Resize(input->dims());

    auto in_dims_vec = phi::vectorize(input->dims());
    in_dims_vec[1] = input->dims()[2];
    in_dims_vec[2] = input->dims()[1];
    transformed_input->Resize(phi::make_ddim(in_dims_vec));
    context.template Alloc<T>(transformed_input);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelFirst(const DeviceContext& context,
                                const Tensor* input,
                                Tensor* transformed_input) {
  VLOG(5) << "Why am I called?";
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    std::vector<int> axis{0, 4, 1, 2, 3};
    phi::funcs::Transpose<DeviceContext, T, 5> trans5;
    trans5(context, *input, transformed_input, axis);

  } else if (dim == 2) {
    std::vector<int> axis{0, 3, 1, 2};
    phi::funcs::Transpose<DeviceContext, T, 4> trans4;
    trans4(context, *input, transformed_input, axis);
  } else if (dim == 1) {
    std::vector<int> axis{0, 2, 1};
    phi::funcs::Transpose<DeviceContext, T, 3> trans3;
    trans3(context, *input, transformed_input, axis);
  }
}

template <typename DeviceContext, typename T>
inline void TransToChannelLast(const DeviceContext& context,
                               const Tensor* input,
                               Tensor* transformed_input) {
  int dim = input->dims().size() - 2;
  if (dim == 3) {
    std::vector<int> axis{0, 2, 3, 4, 1};
    phi::funcs::Transpose<DeviceContext, T, 5> trans5;
    trans5(context, *input, transformed_input, axis);

  } else if (dim == 2) {
    std::vector<int> axis{0, 2, 3, 1};
    phi::funcs::Transpose<DeviceContext, T, 4> trans4;
    trans4(context, *input, transformed_input, axis);
  } else if (dim == 1) {
    std::vector<int> axis{0, 2, 1};
    phi::funcs::Transpose<DeviceContext, T, 3> trans3;
    trans3(context, *input, transformed_input, axis);
  }
}

inline bool IsVoltaOrLater(const phi::GPUContext& dev_ctx) {
  return dev_ctx.GetComputeCapability() >= 70;
}

inline std::vector<int> TransformDimOrder(const std::vector<int>& dims) {
  std::vector<int> transformed_dims(dims.begin(), dims.end());
  if (dims.size() < 4) {
    return transformed_dims;
  }
  int H, W, D, C;
  if (dims.size() == 4) {
    H = dims[1];
    W = dims[2];
    C = dims[3];
    transformed_dims[1] = C;
    transformed_dims[2] = H;
    transformed_dims[3] = W;
  } else {
    D = dims[1];
    H = dims[2];
    W = dims[3];
    C = dims[4];
    transformed_dims[1] = C;
    transformed_dims[2] = D;
    transformed_dims[3] = H;
    transformed_dims[4] = W;
  }
  return transformed_dims;
}

}  // namespace operators
}  // namespace paddle
