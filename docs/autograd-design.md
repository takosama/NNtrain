# Autograd design

This document defines the ownership boundaries introduced in phase 4-1 and the
gradient lifecycle introduced in phase 4-2.

## Responsibilities

| Component | Owns | Does not own |
| --- | --- | --- |
| `Tensor` | Value buffer, gradient buffer, shape, and the public `Backward` facade | Parent-edge storage, graph traversal |
| `AutogradNode` | Immutable parent edges and the node's single local backward action | Tensor values, gradient clearing, graph traversal |
| `AutogradEngine` | Topological traversal, reachable-gradient clearing, output seeding, and reverse execution | Forward numerical operations, optimizer updates |
| Tensor operation | Forward numerical result, result-node creation, and definition of the local derivative | Storage or traversal of the complete graph |

Every tensor has exactly one `AutogradNode`. A public tensor constructor creates
a leaf node with no parents. An operation result creates a node whose parents
are the operation inputs. Parent collections are copied and exposed through a
read-only view.

A node's backward action can be registered exactly once. This prevents an
operation from silently replacing the derivative rule after graph construction.
A leaf without a registered action uses a no-op action during reverse execution.

## Current execution flow

1. A tensor operation computes its forward values.
2. It constructs the result tensor with its input tensors as parents.
3. It registers the result node's local backward action.
4. `Tensor.Backward` delegates to `AutogradEngine`.
5. The engine validates the seed, builds a topological order, clears reachable
   intermediate gradients, accumulates the output seed, and runs each node in
   reverse order.

## Gradient lifecycle

Gradients follow these rules:

1. A leaf is a tensor whose `AutogradNode` has no parents.
2. Leaf gradients accumulate across independent backward traversals.
3. Reachable non-leaf gradients are cleared at the start of every valid backward
   traversal. This prevents stale intermediate gradients from affecting a new
   traversal.
4. The output seed is added to the output gradient. A non-leaf output has just
   been cleared, while a leaf output retains earlier gradients and accumulates
   the new seed.
5. `Tensor.ZeroGrad()` clears only that tensor. `Parameter.ZeroGrad()`,
   `Module.ZeroGrad()`, and `AdamW.ZeroGrad()` explicitly clear parameter leaves.
6. Seed shape is validated before any gradient is cleared or changed. A rejected
   backward request therefore leaves all existing gradients unchanged.

This supports gradient accumulation over multiple independently constructed
loss graphs. Training code must call an appropriate `ZeroGrad` method when it
wants to start a new accumulation window.

## Shared tensors and multiple paths

Graph identity and edge multiplicity follow these rules:

1. Tensor nodes are identified by object reference, not by shape, value, name,
   or value equality.
2. A node's parent list preserves duplicate edges. For example, `x + x` has two
   parent edges that both refer to `x`.
3. Every local backward action adds its contribution with `+=`. Contributions
   from every outgoing branch therefore meet in the shared tensor's gradient.
4. Topological traversal visits each tensor node once by reference identity.
   A shared intermediate's backward action runs only after all downstream
   branches have contributed, and it runs exactly once.
5. Shared leaf gradients continue to follow the phase 4-2 accumulation window
   and are cleared only through an explicit `ZeroGrad`.

This distinction is intentional: edges retain multiplicity for the chain rule,
while nodes are deduplicated for execution.

## Repeated backward

The same retained graph may run `Backward` more than once sequentially:

1. Each call validates its seed and the graph before changing gradients.
2. Reachable non-leaf gradients are cleared and recomputed for that call.
3. Leaf gradients accumulate one new contribution per call.
4. A different valid seed may be supplied on every call.
5. Calling `ZeroGrad` changes only gradients and does not invalidate the graph.
6. Graph nodes and backward actions remain retained as long as the output graph
   remains reachable; there is currently no automatic graph release.

Forward values used by derivative rules must remain unchanged. Tensor data has
an internal version that advances whenever `Parameter.Step`, `Module.Step`, or
`AdamW.Step` updates it. Each `AutogradNode` records its parents' versions during
the forward pass. `Backward` rejects a stale graph before clearing or changing
any gradient and instructs the caller to perform a fresh forward pass.

Repeated calls are defined as sequential operations. Concurrent `Backward`
calls on the same graph are not supported.

## Broadcast gradient reduction

Binary element-wise arithmetic uses one shared execution path:

1. `BinaryBroadcastPlan` accepts equal shapes or a one-element operand and
   determines the output shape and scalar-like sides.
2. The shared executor computes forward values and constructs the graph node.
3. Each operation supplies only its forward formula and its local derivatives
   with respect to the left and right values.
4. For a non-scalar operand, the executor adds one contribution per output
   element to the corresponding gradient element.
5. For a scalar-like operand, it sums contributions from every output element
   and adds the reduced value to gradient element zero.
6. Left and right reductions are independent. If both edges refer to the same
   Tensor, both derivative contributions are retained.

This path is used by addition, subtraction, multiplication, and division.
General dimension broadcasting remains unsupported.

## Topological traversal

`AutogradEngine` builds post-order iteratively rather than with recursive DFS:

1. A `TraversalFrame` records a Tensor and whether its parents have been
   expanded.
2. The engine uses an explicit `Stack<TraversalFrame>` allocated on the heap.
3. A Tensor is marked visited by reference identity when its unexpanded frame
   is processed.
4. The expanded frame is pushed before its parents. Parents are pushed in
   reverse order, preserving the same deterministic processing order as the
   previous recursive implementation.
5. The Tensor is appended to topological order when its expanded frame is
   popped.

Traversal time is `O(V + E)`. The visited set, result list, and explicit stack
use `O(V + E)` heap memory in the worst case. Graph depth no longer consumes the
thread call stack, so deeply chained graphs do not cause recursive stack
overflow. Extremely large graphs can still exhaust available heap memory.

Autograd edges are immutable and can only point to tensors that already exist,
so graphs constructed through Tensor operations are directed acyclic graphs.
The traversal does not perform a separate cycle-detection pass.

## NoGrad forward scopes

`AutogradContext.NoGrad()` disables graph recording for forward operations until
its returned scope is disposed:

1. Forward numerical values and shapes are computed normally.
2. Operation results receive a detached leaf node with no parent edges.
3. A detached node does not retain the operation's local backward action.
4. Calling `Backward` on a detached result seeds that result's own gradient but
   cannot propagate to tensors used inside the `NoGrad` scope.
5. A detached result used by tracked operations after the scope behaves as a new
   leaf. Gradients stop at that boundary.
6. `NoGrad` affects forward graph construction only. It does not disable
   `Backward` on graphs that were already recorded.
7. Scopes are nestable and recording resumes after the outermost scope is
   disposed, including exception paths.
8. Scope depth is stored in `AsyncLocal`, so the state follows the logical async
   execution context without becoming a process-wide global switch.
9. Detached operation results allocate gradient storage lazily. If a detached
   result is later used by a recorded operation, its leaf gradient buffer is
   created before that graph is built.

Callers must dispose the scope, normally with `using`. Concurrent forward work
can use independent execution contexts. Concurrent mutation or `Backward` on
the same Tensor graph remains unsupported.
