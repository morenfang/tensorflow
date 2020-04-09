/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/gtl/flatset.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

constexpr char kArithmeticOptimizer[] = "ArithmeticOptimizer";

// Optimize TF computations by reducing the arithmetic complexity required to
// run a model.
class ArithmeticOptimizer : public GraphOptimizer {
 public:
  ArithmeticOptimizer()
      : opt_level_(RewriterConfig::ON),
        options_(ArithmeticOptimizerOptions::Default(RewriterConfig::ON)) {}

  explicit ArithmeticOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level),
        options_(ArithmeticOptimizerOptions::Default(opt_level)) {}

  ~ArithmeticOptimizer() override {}

  string name() const override { return "arithmetic_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  friend class ArithmeticOptimizerTest;

  // Granular control for arithmetic optimizer stages
  struct ArithmeticOptimizerOptions {
    bool combine_add_to_addn = false;
    bool convert_sqrt_div_to_rsqrt_mul = false;
    bool dedup_computations = false;
    bool fold_conjugate_into_transpose = false;
    bool fold_multiply_into_conv = false;
    bool fold_transpose_into_matmul = false;
    bool fuse_squared_diff = false;
    bool hoist_common_factor_out_of_aggregation = false;
    bool hoist_concat_chains = true;
    bool hoist_cwise_unary_chains = false;
    bool minimize_broadcasts = false;
    bool optimize_max_or_min_of_monotonic = false;
    bool remove_idempotent = false;
    bool remove_identity_transpose = false;
    bool remove_involution = false;
    bool remove_logical_not = false;
    bool remove_negation = false;
    bool remove_redundant_bitcast = false;
    bool remove_redundant_cast = false;
    bool remove_redundant_reshape = false;
    bool reorder_cast_like_and_value_preserving = false;
    bool replace_mul_with_square = false;
    bool simplify_aggregation = false;
    bool convert_pow = false;
    bool convert_log1p = false;
    bool convert_log_softmax = false;
    bool convert_expm1 = false;
    bool unary_ops_composition = false;
    bool remove_stack_strided_slice_same_axis = false;

    // Choose which arithmetic optimizer stages will be enabled for a given
    // optimization level by default.
    static ArithmeticOptimizerOptions Default(
        RewriterConfig::Toggle opt_level) {
      ArithmeticOptimizerOptions options;
      return options;
    }
  };

  // Returns true if it is safe to dedup node from the graph.
  bool CanDedup(const NodeDef& node) const;

  // Dedup redundant nodes in the graph.
  void DedupComputations();

  // Forward the control dependencies anchored on src_nodes to the target_nodes.
  void ForwardControlDependencies(NodeDef* target_node,
                                  const std::vector<const NodeDef*>& src_nodes);

  // Runs peep-hole optimizations on `optimized_graph`, e.g., removing inverse
  // transposes.
  Status SimplifyArithmeticOps(bool can_use_shapes);
  // Tries to simplify the expression that roots at `node` and replaces the uses
  // of `node` to the simplified expression. Returns the name of the simplified
  // tensor (e.g. "split:1") or an empty string if no simplification is
  // performed.
  //
  // `node_map` stores the mapping from node names to NodeDef*, and will be
  // updated according to the rewrite.
  //
  // `new_nodes` will be populated with the new nodes this function creates and
  // updates. The caller can push these nodes into the simplification queue to
  // optimize them further.
  //
  // TODO(jingyue): This interface is not suitable for optimizing nodes with
  // multiple output tensors. We should pass in a tensor name instead of a
  // NodeDef.
  string TrySimplifyAndReplaceUses(const NodeDef* node,
                                   SetVector<NodeDef*>* nodes_to_simplify);

  RewriterConfig::Toggle opt_level_;
  ArithmeticOptimizerOptions options_;

  bool fetch_nodes_known_ = false;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<NodeMap> node_map_;
  std::unique_ptr<GraphProperties> graph_properties_;
  GraphDef* optimized_graph_ = nullptr;  // Not owned.
  gtl::FlatSet<string> feed_nodes_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_
