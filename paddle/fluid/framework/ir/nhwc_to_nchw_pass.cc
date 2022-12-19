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

#include "paddle/fluid/framework/ir/nhwc_to_nchw_pass.h"

#include "paddle/fluid/framework/data_layout_transform.h"

namespace paddle {
namespace framework {
namespace ir {

void NHWC2NCHWPass::ApplyImpl(Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::PreconditionNotMet(
                              "During the nhwc_to_nchw_pass, the graph "
                              "should not be null."));
  FusePassBase::Init("nhwc_to_nchw_pass", graph);

  auto *scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::PreconditionNotMet(
                              "During the nhwc_to_nchw_pass, the scope "
                              "should not be null."));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
