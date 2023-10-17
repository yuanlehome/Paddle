// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/instruction/cinn_jit_instruction.h"

#include "paddle/cinn/hlir/dialect/runtime/ir/jit_kernel_op.h"
#include "paddle/cinn/hlir/dialect/runtime/ir/runtime_dialect.h"
#include "paddle/cinn/hlir/framework/instruction.h"
#include "paddle/cinn/hlir/framework/new_ir_compiler.h"
#include "paddle/cinn/runtime/cuda/cuda_util.h"
#include "paddle/fluid/framework/new_executor/pir_adaptor/pir_adaptor_util.h"
#include "paddle/fluid/framework/paddle2cinn/transform_type.h"

namespace paddle {
namespace framework {

class CinnJitInstruction::FnPtrImpl {
  using CUDAJITInfo = cinn::hlir::framework::newir::CUDAJITInfo;

 public:
  explicit FnPtrImpl(const CUDAJITInfo& cuda_jit_info)
      : cuda_jit_info_(cuda_jit_info) {}
  // TODO(Aurelus84): Support to specify name2podargs and stream arguments.
  void Run(const std::vector<phi::DenseTensor*>& kernel_args, void* stream) {
    auto fn = static_cast<CUfunction>(cuda_jit_info_.fn_ptr);

    auto stream1 = static_cast<CUstream>(stream);

    pass_arg.clear();
    vec_temp_.resize(kernel_args.size());
    for (size_t i = 0; i < kernel_args.size(); ++i) {
      vec_temp_[i] = kernel_args[i]->data();
      pass_arg.push_back(vec_temp_.data() + i);
    }

    CUDA_DRIVER_CALL(cuLaunchKernel(fn,
                                    cuda_jit_info_.grid_dims[0],
                                    cuda_jit_info_.grid_dims[1],
                                    cuda_jit_info_.grid_dims[2],
                                    cuda_jit_info_.block_dims[0],
                                    cuda_jit_info_.block_dims[1],
                                    cuda_jit_info_.block_dims[2],
                                    0,  // share memory
                                    stream1,
                                    pass_arg.data(),
                                    nullptr))
  }

 private:
  CUDAJITInfo cuda_jit_info_;

  std::vector<void*> vec_temp_;
  std::vector<void*> pass_arg;
};

CinnJitInstruction::CinnJitInstruction(
    size_t id,
    const platform::Place& place,
    ::pir::Operation* op,
    const ValueExecutionInfo& value_exec_info)
    : InstructionBase(id, place) {
  // TODO(Aurelius84): We shall simplify members of JitKernelOp to make it
  // only hold related function ptrs. Impl is the real runtime data structure
  // responsible to construct hlir::framework::Instruction.
  auto jit_kernel_op = op->dyn_cast<cinn::dialect::JitKernelOp>();
  fn_ptr_impl_ = std::make_shared<FnPtrImpl>(jit_kernel_op.cuda_jit_info());
  op_ = op;

  place_ = place;

  InitInputsOutputsIds(op, value_exec_info);

  for (size_t i = 0; i < op->num_operands(); ++i) {
    auto in = op->operand_source(i);

    auto var_name = value_exec_info.GetVarName(in);

    auto tensor = value_exec_info.GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args.push_back(tensor);
  }

  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);

  for (size_t i = 0; i < op->num_results(); ++i) {
    pir::Value result = op->result(i);
    auto var_name = value_exec_info.GetVarName(result);

    auto tensor = value_exec_info.GetScope()
                      ->Var(var_name)
                      ->GetMutable<phi::DenseTensor>();

    tensor_args.push_back(tensor);

    out_tensor_ = tensor;

    auto alloc_tensor_type =
        result.type().dyn_cast<paddle::dialect::AllocatedDenseTensorType>();
    tensor->set_type(
        paddle::dialect::TransToPhiDataType(alloc_tensor_type.dtype()));
    tensor->Resize(alloc_tensor_type.dims());
  }
}

void CinnJitInstruction::Run() {
  // VLOG(6) << "Run cinn jit_kernel_op : " << Name();
  // Get kernel input
  auto gpu_ctx = static_cast<phi::GPUContext*>(dev_ctx_);

  //  gpu_ctx->Wait();
  auto stream = gpu_ctx->stream();
  for (size_t i = 0; i < tensor_args.size(); ++i) {
    gpu_ctx->Alloc(tensor_args[i], phi::DataType::FLOAT32);
  }

  fn_ptr_impl_->Run(tensor_args, static_cast<void*>(stream));
}

const std::string& CinnJitInstruction::Name() const {
  // TODO(Aurelius84): Consider the case for instrucitons constaning
  // multipule function ptrs and function names.
  // return impl_->pointer()->function_name();
  static const std::string name = "cinn_jit";
  return name;
}

}  // namespace framework
}  // namespace paddle
