# Module design

## Parameter enumeration

Phase 5-1 centralizes parameter enumeration in the `Module` base class.
Concrete modules no longer override `Parameters()`.

Subclasses build their structure with two protected methods:

- `RegisterParameter` registers a direct `Parameter`.
- `RegisterModule` registers a child `Module`.

Both methods return the registered object so constructors can assign a property
and register it in one expression.

`Module` stores parameters and child modules in one ordered member list.
`Parameters()` expands that list lazily:

1. A direct parameter is yielded immediately.
2. A child module is expanded depth-first through the same base implementation.
3. Parameter/module interleaving is preserved.
4. Repeated enumeration returns the same parameter references in the same order.

Stable order matters because optimizers associate state with the enumerated
parameters. The refactor preserves the previous order, including
`TransformerClassifier`'s `Pos`, blocks, and classification head ordering.

## Parameter metadata

Phase 5-2 gives every `Parameter` explicit metadata:

- `Name` is required and cannot be empty or whitespace. The same local name is
  assigned to its Tensor.
- `Owner` is the direct Module that called `RegisterParameter`. Parameters in a
  child module remain owned by that child rather than by the root model.
- A standalone Parameter may remain unowned for low-level optimizer use.
- Ownership is immutable once assigned. Registration by a different Module is
  rejected rather than silently replacing the owner.
- `WeightDecay` is either `Apply` or `Exclude`; it is no longer inferred solely
  from Tensor rank.

The built-in policy is:

| Parameter | Policy |
| --- | --- |
| Linear weight (`W`) | `Apply` |
| Linear bias (`B`) | `Exclude` |
| LayerNorm `Gamma` and `Beta` | `Exclude` |
| Transformer positional parameter (`Pos`) | `Apply` |

AdamW consumes this policy directly. `AdamWOptions.Decay1D` remains available
as a compatibility override for one-dimensional parameters.

## Optimizer-only parameter updates

Phase 5-3 removes `Step` from both `Module` and `Parameter`. Public training code
can clear gradients through those objects, but it cannot update parameter data
through them.

Parameter data mutation is exposed only as an internal, versioned update scope
for optimizer implementations. AdamW acquires that scope from each Parameter,
applies its update, and disposes the scope to advance the Tensor data version.
This keeps stale-graph detection intact while making the optimizer the single
public owner of learning updates.

## Optimizer abstraction

Phase 5-4 introduces `IOptimizer` as the training loop's optimizer contract.
The contract contains only the two lifecycle operations common to optimizer
implementations:

- `ZeroGrad()` clears gradients for all parameters managed by the optimizer.
- `Step()` applies one update using the currently accumulated gradients.

AdamW implements this contract, and the application entry point stores it as an
`IOptimizer`. Algorithm-specific configuration remains on AdamW rather than
leaking into the common interface.

## AdamW options

Phase 5-5 moves every AdamW hyperparameter into the immutable
`AdamWOptions` value object. AdamW now accepts only its managed parameters and
an optional options object; omitting the object retains the previous defaults.

| Option | Default |
| --- | --- |
| `LearningRate` | `1e-3` |
| `Beta1` | `0.9` |
| `Beta2` | `0.999` |
| `Epsilon` | `1e-8` |
| `WeightDecay` | `1e-2` |
| `Decay1D` | `false` |

Configuration remains specific to AdamW and is deliberately absent from
`IOptimizer`. Adding another optimizer therefore does not expand the common
lifecycle contract.

## Duplicate parameter detection

Phase 5-6 rejects duplicate registration by reference identity. Parameters with
the same name remain valid when they are different objects.

Detection occurs at three boundaries:

1. `Module.RegisterParameter` rejects the same direct Parameter twice.
2. `Module.RegisterModule` rejects the same direct child twice, while recursive
   parameter enumeration rejects shared children and parameters reached through
   multiple paths.
3. AdamW materializes and validates its input before allocating optimizer state;
   supplying the same Parameter more than once throws instead of updating it
   twice.

`Module.ZeroGrad()` completes recursive validation before clearing any gradient,
so an invalid module graph cannot cause a partial clear.

## Optimizer state

Phase 5-7 replaces AdamW's Parameter-keyed state dictionaries with a versioned,
ordered `AdamWState` structure. A state snapshot contains:

- the state format version and completed step count;
- the AdamW options needed to continue with the same update rule;
- one slot per Parameter containing its stable index, name, shape, first moment,
  and second moment.

Slots follow the stable `Module.Parameters()` order established in phase 5-1.
Names do not need to be globally unique because index and shape are validated
together. Parameter object references are not stored in the snapshot.

`CaptureState()` returns a deep copy suitable for serialization.
`RestoreState()` validates the format version and complete Parameter layout
before replacing the live optimizer state, and also makes a defensive copy.
The model's Parameter data is a separate checkpoint concern and must be restored
before the optimizer state.
