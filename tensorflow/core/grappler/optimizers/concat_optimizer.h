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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/custom_graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

static const char* kConcatOptimizerName = "ConcatOptimizer";

class ConcatOptimizer : public CustomGraphOptimizer {
 public:
  Status Init(const tensorflow::RewriterConfig_CustomGraphOptimizer* config =
                  nullptr) override {
    return Status::OK();
  }
  string name() const override { return "concat_optimizer"; };
  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;
  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;
 private:
  std::unique_ptr<NodeMap> node_map_;
  std::unique_ptr<GraphProperties> graph_properties_;
  GraphDef* optimized_graph_ = nullptr;  // Not owned.
  gtl::FlatSet<string> feed_nodes_;
};
}
}

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONCAT_OPTIMIZER_H_