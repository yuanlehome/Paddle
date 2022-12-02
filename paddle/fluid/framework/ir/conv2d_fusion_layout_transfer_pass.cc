// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/conv2d_fusion_layout_transfer_pass.h"
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void Conv2dFusionLayoutTransferPass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::PreconditionNotMet("graph should not be nullptr."));
  FusePassBase::Init("conv2d_fusion_layout_transfer", graph);
  auto *scope = param_scope();

  PADDLE_ENFORCE_EQ(graph->IsMainGraph(),
                    true,
                    platform::errors::InvalidArgument(
                        "the graph should be main graph when applying "
                        "conv2d_fusion layout transfer pass"));

  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal("scope must not be nullptr when applying "
                              "conv2d_fusion layout transfer pass"));

  std::unordered_map<framework::ir::Node *, framework::ir::Node *> cache;
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  auto iter = op_nodes.cbegin();
  auto *block_desc = (*iter)->Op()->Block();

  for (auto *node : op_nodes) {
    if (node->IsOp() && node->Name() == "conv2d_fusion") {
      auto *op_desc = node->Op();
      auto &&data_format = op_desc->GetAttrIfExists<std::string>("data_format");
      if (data_format == "NCHW") {
        for (auto *in_node : node->inputs) {
          if (!in_node->IsVar() || in_node->Var()->Persistable()) continue;
          if (in_node->inputs[0]->Name() == "conv2d_fusion") continue;
          insertLayoutTransOp(graph,
                              in_node,
                              node,
                              framework::DataLayout::kNCHW,
                              framework::DataLayout::kNHWC,
                              block_desc,
                              &cache);
        }

        auto nhwc_attr = framework::Attribute(std::string("NHWC"));
        op_desc->SetAttr("data_format", nhwc_attr);

        for (auto *out_node : node->outputs) {
          if (!out_node->IsVar() || out_node->Var()->Persistable()) continue;
          auto from_shape = out_node->Var()->GetShape();
          out_node->Var()->SetShape(
              {from_shape[0], from_shape[2], from_shape[3], from_shape[1]});
          if (out_node->outputs[0]->Name() == "conv2d_fusion") continue;
          insertLayoutTransOp(graph,
                              out_node,
                              out_node->outputs[0],
                              framework::DataLayout::kNHWC,
                              framework::DataLayout::kNCHW,
                              block_desc,
                              &cache);
        }
      }
    }
  }
}

void Conv2dFusionLayoutTransferPass::insertLayoutTransOp(
    framework::ir::Graph *graph,
    framework::ir::Node *prev_node,
    framework::ir::Node *next_node,
    framework::DataLayout from_layout,
    framework::DataLayout to_layout,
    framework::BlockDesc *block_desc,
    std::unordered_map<framework::ir::Node *, framework::ir::Node *> *cache)
    const {
  auto do_insert = [&](const std::string &in_var_name,
                       const std::string &out_var_name) {
    auto update_op_desc = [&](framework::OpDesc &desc,
                              const std::string &x_name,
                              const std::string &out_name) {
      desc.SetType("transfer_layout");
      desc.SetInput("X", {x_name});
      desc.SetOutput("Out", {out_name});
      desc.SetAttr("src_layout", static_cast<int>(from_layout));
      desc.SetAttr("dst_layout", static_cast<int>(to_layout));
      desc.Flush();
    };
    CHECK_NOTNULL(block_desc);
    if (cache->count(prev_node) == 0) {
      framework::OpDesc op_desc(block_desc);
      update_op_desc(op_desc, in_var_name, out_var_name);
      auto *op_node = graph->CreateOpNode(&op_desc);
      auto *op_out_var_desc = block_desc->Var(out_var_name);

      op_out_var_desc->SetPersistable(false);
      op_out_var_desc->SetDataType(prev_node->Var()->GetDataType());
      auto to_shape = prev_node->Var()->GetShape();
      if (from_layout == framework::DataLayout::kNCHW) {
        auto n = to_shape[0];
        auto c = to_shape[1];
        auto h = to_shape[2];
        auto w = to_shape[3];
        op_out_var_desc->SetShape({n, h, w, c});
      } else {
        auto n = to_shape[0];
        auto h = to_shape[1];
        auto w = to_shape[2];
        auto c = to_shape[3];
        op_out_var_desc->SetShape({n, c, h, w});
      }

      auto *op_out_var_node = graph->CreateVarNode(op_out_var_desc);
      IR_NODE_LINK_TO(op_node, op_out_var_node);
      cache->insert(std::make_pair(prev_node, op_out_var_node));
    }
    next_node->Op()->RenameInput(prev_node->Name(),
                                 cache->at(prev_node)->Name());
    IR_NODE_LINK_TO(prev_node, cache->at(prev_node)->inputs.front());
    IR_NODE_LINK_TO(cache->at(prev_node), next_node);
  };

  if (from_layout == framework::DataLayout::kNCHW &&
      to_layout == framework::DataLayout::kNHWC) {
    auto in_var_name = prev_node->Var()->Name();
    auto out_var_name = in_var_name + "_nchw_to_nhwc";
    do_insert(in_var_name, out_var_name);

  } else if (from_layout == framework::DataLayout::kNHWC &&
             to_layout == framework::DataLayout::kNCHW) {
    auto in_var_name = prev_node->Var()->Name();
    auto out_var_name = in_var_name + "_nhwc_to_nchw";
    do_insert(in_var_name, out_var_name);
  } else {
    //
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv2d_fusion_layout_transfer_pass,
              paddle::framework::ir::Conv2dFusionLayoutTransferPass);
