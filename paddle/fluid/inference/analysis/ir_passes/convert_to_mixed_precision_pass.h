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

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/analysis/argument.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/backend.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {

using VarType = framework::proto::VarType;
using BlockID = size_t;

class ConvertToMixedPrecisionPass : public framework::ir::Pass {
 public:
  ConvertToMixedPrecisionPass() = default;

  explicit ConvertToMixedPrecisionPass(
      const std::string& model_file,
      const std::string& params_file,
      const std::string& mixed_model_file,
      const std::string& mixed_params_file,
      phi::DataType mixed_precision,
      phi::Backend backend,
      bool keep_io_types,
      const std::unordered_set<std::string>& black_list)
      : model_file_(model_file),
        params_file_(params_file),
        mixed_model_file_(mixed_model_file),
        mixed_params_file_(mixed_params_file),
        mixed_precision_(mixed_precision),
        backend_(backend),
        keep_io_types_(keep_io_types),
        black_list_(black_list) {}

  // Used for convert_to_mixed_precision interface
  void Run();

 protected:
  // Used for analysis pass convert_to_mixed_precision_pass
  void ApplyImpl(framework::ir::Graph* main_graph) const override;

 private:
  void LoadAndPrepare();
  void ConvertAllFp64ToFp32(framework::ir::Graph* graph) const;
  void SaveMixedModel();
  void ConvertTensorDtype(BlockID block_idx) const;
  void ProcessInputNode(bool support_precision,
                        framework::ir::Node* in_node,
                        framework::ir::Node* op_node,
                        int* suffix,
                        framework::BlockDesc* block_desc,
                        VarType::Type to_type,
                        BlockID block_idx) const;

  void ProcessOutputNode(BlockID block_idx,
                         framework::ir::Node* var_node,
                         VarType::Type to_type) const;

  // To support multi block, we need to consider a lot of special cases.
  // Return Node* which first appers in block.
  framework::ir::Node* GetRealVarNode(BlockID block_idx,
                                      framework::ir::Node* node) const;
  void FindVarsInMultiBlock() const;
  bool VarIsMultiPrecisionOpsOut(BlockID block_idx,
                                 framework::ir::Node* op_node) const;

 private:
  // A trick. Patch for strange op, which input name equal to output name, such
  // as `fused_multi_transformer`
  void PatchForStrangeOp() const;

 private:
  std::string model_file_;
  std::string params_file_;
  std::string mixed_model_file_;
  std::string mixed_params_file_;
  mutable phi::DataType mixed_precision_;
  mutable phi::Backend backend_;
  mutable bool keep_io_types_;
  mutable std::unordered_set<std::string> black_list_;
  framework::Executor executor_{platform::CPUPlace()};
  framework::Scope scope_;

  mutable std::unordered_map<framework::ir::Node*, framework::ir::Node*>
      cast_map_;
  mutable std::unordered_map<std::string, std::pair<VarType::Type, BlockID>>
      vars_in_multi_block_with_pair_;
  mutable std::unordered_map<std::string, std::vector<std::string>>
      vars_in_multi_block_with_ops_;
  mutable int suffix_{0};

  std::unique_ptr<framework::ir::Graph> main_graph_{nullptr};
  mutable std::vector<framework::ir::Graph*> graphes_;
};

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist);

void AddCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* node,
    framework::ir::Node* next_op,
    VarType::Type from_type,
    VarType::Type to_type,
    int* suffix,
    framework::BlockDesc* block_desc,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* map);

void ConvertToMixedPrecision(const std::string& model_file,
                             const std::string& params_file,
                             const std::string& mixed_model_file,
                             const std::string& mixed_params_file,
                             phi::DataType mixed_precision,
                             phi::Backend backend,
                             bool keep_io_types,
                             const std::unordered_set<std::string>& black_list);

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
