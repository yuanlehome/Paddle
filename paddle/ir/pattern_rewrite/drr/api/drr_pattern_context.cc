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

#include "paddle/ir/pattern_rewrite/drr/api/drr_pattern_context.h"

#include <glog/logging.h>
#include "paddle/ir/pattern_rewrite/drr/pattern_graph.h"

namespace ir {
namespace drr {

DrrPatternContext::DrrPatternContext() {
  source_pattern_graph_ = std::make_shared<SourcePatternGraph>();
  result_pattern_graph_ = std::make_shared<ResultPatternGraph>();
}

drr::SourcePattern DrrPatternContext::SourcePattern() {
  return drr::SourcePattern(this);
}
const Op& DrrPatternContext::SourceOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes) {
  owned_ops_.push_back(std::shared_ptr<drr::Op>(
      new drr::Op(op_type, attributes, source_pattern_graph_.get())));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPatternContext::SourceTensorPattern(
    const std::string& tensor_id) {
  return source_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(tensor_id, source_pattern_graph_.get())));
}

const Op& DrrPatternContext::ResultOpPattern(
    const std::string& op_type,
    const std::unordered_map<std::string, Attribute>& attributes) {
  owned_ops_.push_back(std::shared_ptr<drr::Op>(
      new drr::Op(op_type, attributes, result_pattern_graph_.get())));
  return *owned_ops_.back();
}

const drr::Tensor& DrrPatternContext::ResultTensorPattern(
    const std::string& tensor_id) {
  return result_pattern_graph_->AddTensor(std::shared_ptr<drr::Tensor>(
      new drr::Tensor(tensor_id, result_pattern_graph_.get())));
}

std::vector<Constraint> DrrPatternContext::constraints() const {
  return constraints_;
}

// void DrrPatternContext::RequireEqual(const Attribute& first, const Attribute&
// second) {
//   auto constrain_fn = [&](const MatchContext& match_context) {
//     return match_context.Attr(first.id()) == match_context.Attr(second.id());
//   };
//   constraints_.emplace_back(constrain_fn);
// }

void DrrPatternContext::RequireEqual(const TensorShape& first,
                                     const TensorShape& second) {
  auto constrain_fn = [&](const MatchContext& match_context) {
    return match_context.Tensor(first.tensor_id()).Shape() ==
           match_context.Tensor(second.tensor_id()).Shape();
  };
  constraints_.emplace_back(constrain_fn);
}

void Op::operator()(const Tensor& arg, const Tensor* out) const {
  std::vector<const Tensor*> inputs{arg.shared_from_this()};
  std::vector<const Tensor*> outputs{out->shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
}

Tensor& Op::operator()(const Tensor& arg) const {
  std::vector<const Tensor*> inputs{arg.shared_from_this()};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      "tmp_" + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{out.shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
  return out;
}

Tensor& Op::operator()() const {
  std::vector<const Tensor*> inputs{};
  auto& out = pattern_graph_->AddTmpTensor(std::shared_ptr<Tensor>(new Tensor(
      "tmp_" + op_type_name_ + "_" + std::to_string(count++), pattern_graph_)));
  std::vector<const Tensor*> outputs{out.shared_from_this()};
  pattern_graph_->AddOpCall(
      std::make_shared<OpCall>(shared_from_this(), inputs, outputs));
  return out;
}

int64_t Op::count = 0;

void Tensor::operator=(Tensor& other) const {  // NOLINT
  // The two tensor must be in the same pattern graph.
  CHECK(this->pattern_graph_ == other.pattern_graph_);
  if (other.tensor_id_.substr(0, 4) == "tmp_" &&
      tensor_id_.substr(0, 4) != "tmp_") {
    other.pattern_graph_->UpdateTmpTensor(other.tensor_id_, this->tensor_id_);
  }
}

}  // namespace drr
}  // namespace ir
