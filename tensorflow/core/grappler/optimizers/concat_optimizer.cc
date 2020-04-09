/* Copyright 2020 Enflame corp. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/grappler/optimizers/concat_optimizer.h"

#include <algorithm>
#include <deque>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/graph_topology_view.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/symbolic_shapes.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/tensor_coding.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/saved_tensor_slice_util.h"
#include "tensorflow/core/util/strided_slice_op.h"

using tensorflow::str_util::StringReplace;
using tensorflow::strings::StrCat;

namespace tensorflow {
namespace grappler {
namespace {

bool IsSupport(const NodeDef* node) {
  if (IsConcat(*node) && node->attr().count("N") != 0) {
    const int n = node->attr().at("N").i();
    return n > 1;
  } else {
    return false;
  }
}

}  // namespace

Status ConcatOptimizer::Optimize(Cluster* cluster,
                                 const GrapplerItem& item,
                                 GraphDef* optimized_graph) {
  VLOG(2) << "Custom optimizer of ConcatOptimizer. Optimize() Entering.";
  SetVector<NodeDef*> concat_nodes;
  GrapplerItem optimized_item(item);
  optimized_graph_ = &optimized_item.graph;
    
  node_map_.reset(new NodeMap(optimized_graph_));
  for (const auto& feed : item.feed) {
    feed_nodes_.insert(NodeName(feed.first));
    VLOG(2) << "Feed node name: " << NodeName(feed.first);
  }
  TF_RETURN_IF_ERROR(TopologicalSort(optimized_graph_));
  GRAPPLER_RETURN_IF_DEADLINE_EXCEEDED();
    
  for (int i = 0; i < optimized_graph_->node_size(); ++i) {
    NodeDef* node = optimized_graph_->mutable_node(i);
    if (IsSupport(node)) {
      concat_nodes.PushBack(node);
      VLOG(2) << "Find Concat node: "
              << node->name()
              << ". N=" << node->attr().at("N").i();
    }
  }

  optimized_graph->Swap(&optimized_item.graph);
  return Status::OK();
}
void ConcatOptimizer::Feedback(Cluster* cluster,
                               const GrapplerItem& item,
                               const GraphDef& optimized_graph,
                               double result) {
  // Nothing to du for ConcatOptimizer.
}

}  // namespace grappler
}  // namespace tensorflow