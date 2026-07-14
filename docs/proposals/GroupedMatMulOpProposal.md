<!--
Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT
-->

# RFC: GroupedMatMul — Proposal for ONNX Standard

**Status:** Proposal  
**ONNX Discussion:** *(to be filed as a GitHub Discussion in onnx/onnx)*  
**Related:** [onnx/onnx#7902](https://github.com/onnx/onnx/issues/7902)  
**Reference implementation:** `com.microsoft.GroupedMatMul` (ONNX Runtime `contrib_ops`, opset 1)

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Prior Art and Framework Support](#2-prior-art-and-framework-support)
3. [Why Not a Function?](#3-why-not-a-function)
4. [Operator Specification](#4-operator-specification)
   - 4.1 [Name and Domain](#41-name-and-domain)
   - 4.2 [Inputs](#42-inputs)
   - 4.3 [Outputs](#43-outputs)
   - 4.4 [Type Constraints](#44-type-constraints)
   - 4.5 [Attributes](#45-attributes)
   - 4.6 [Shape Inference Rules](#46-shape-inference-rules)
5. [Semantics](#5-semantics)
   - 5.1 [Pseudocode](#51-pseudocode)
   - 5.2 [Edge Cases](#52-edge-cases)
6. [Reference Implementation (Python)](#6-reference-implementation-python)
7. [Test Cases](#7-test-cases)
8. [Typical Usage — MoE Feed-Forward Layer](#8-typical-usage--moe-feed-forward-layer)
9. [Design Alternatives Considered](#9-design-alternatives-considered)
10. [Relationship to Existing ONNX Operators](#10-relationship-to-existing-onnx-operators)
11. [Files Required for onnx/onnx PR](#11-files-required-for-onnxonnx-pr)

---

## 1. Motivation

Mixture-of-Experts (MoE) feed-forward layers are a central component of large language models (Mixtral, DeepSeek, Grok, Switch Transformer, etc.). The core computation of an MoE layer is:

> Given a batch of `M` token vectors and a set of `num_groups` expert weight matrices, multiply each token by one or more expert matrices chosen per-token by a router. Optionally, combine the `k` per-expert results with learned combine weights.

This pattern — often called **grouped matrix multiplication** or **grouped GEMM** — can be expressed using existing standard ONNX operators, but a straightforward (unfused) implementation will be very inefficient and impractical.

* The natural decomposition (`Gather` weights → `Expand` tokens → batched `MatMul`) materialises a full `[M×k, K, N]` weight slice and an `[M×k, K]` copy of tokens. For real MoE layers these tensors are gigabytes in size, making the decomposition impractical.
* A fused grouped-GEMM kernel, on the other hand, processes each expert weight matrix once regardless of the number of tokens that select it, and reuses each token row across its `k` experts without copying. It can additionally fuse the `k`-way weighted sum (the "combine" step), avoiding a materialised `[M, k, N]` intermediate.


Adding `GroupedMatMul` to the ONNX standard will enable a more compact and efficient representation of MoE models.

---

## 2. Prior Art and Framework Support

| Framework / Library | API |
|---|---|
| PyTorch | `torch.nn.functional.grouped_mm` (PyTorch ≥ 2.5) |
| PyTorch | `torch._grouped_mm` / `torch.ops.aten.mm_group` (internal) |
| JAX | `jax.lax.dot_general` with grouped batching |
| cuBLAS | `cublasGemmBatchedEx` / `cublasGemmGroupedBatchedEx` |
| CUTLASS | `GroupedGemm` kernel |
| OpenVINO | `GroupConvolution` (analogous for convolution) |

The PyTorch `torch.nn.functional.grouped_mm` API (added in 2.5) directly matches the semantics proposed here:

```python
# PyTorch grouped_mm — same semantics
out = torch.nn.functional.grouped_mm(input, weight, offs=None)
# offs are contiguous group offsets; our design uses indices instead (see §9.3)
```

---

## 3. Function Decomposition

The reference decomposition is:

```
# Per-expert results r: [M, k, N]
idx_flat  = Reshape(group_indices, [M*k])
W_sel     = Gather(weights, idx_flat, axis=0)            # [M*k, K, N] — duplicates weights!
X         = Reshape(Expand(Unsqueeze(input,1),[M,k,K]),  # [M*k, 1, K] — copies tokens!
                    [M*k, 1, K])
r         = Reshape(MatMul(X, W_sel), [M, k, N])
# combine present:
output    = ReduceSum(r * Unsqueeze(combine_weights,-1), axis=1)   # [M, N]
```

---

## 4. Operator Specification

### 4.1 Name and Domain

| Field | Value |
|---|---|
| **Name** | `GroupedMatMul` |
| **Domain** | `ai.onnx` (standard) |
| **Opset version** | Next available opset (e.g. 27) |
| **Since version** | (new in this opset) |

### 4.2 Inputs

| Index | Name | Type | Required | Shape | Description |
|---|---|---|---|---|---|
| 0 | `input` | T | **Required** | `[M, K]` | Row-major token matrix. `M` tokens, `K` is the contraction (hidden) dimension. |
| 1 | `weights` | T | **Required** | `[G, K, N]` | Stack of `G` expert weight matrices, each `K × N`. All experts share the same `K` and `N`. |
| 2 | `group_indices` | tensor(int64) | **Required** | `[M, k]` | Group (expert) index per token per slot. Each of the `M` tokens selects `k` experts. Values must be in `[0, G)`. Use `k=1` for the dense (single-expert) case. |
| 3 | `combine_weights` | T | **Optional** | `[M, k]` | Per-selection combine weight. When present, the output is the weighted sum over the `k` selected experts (shape `[M, N]`). When absent, per-expert results are returned (shape `[M, k, N]`). |
| 4 | `bias` | T | **Optional** | `[G, N]` | Per-group bias vector. Added to each expert's result before the optional combine. |

**Notes:**

* `G = weights.shape[0]` (number of groups / experts).
* Callers with batched inputs of shape `[B, M, K]` should `Reshape` the batch dimensions into `M` first. In ONNX Runtime this is a zero-copy metadata-only view.
* `weights` and `bias` are the same for all tokens (i.e., they are model parameters, not per-token).

### 4.3 Outputs

| Index | Name | Type | Shape | Description |
|---|---|---|---|---|
| 0 | `output` | T | `[M, N]` or `[M, k, N]` | When `combine_weights` is present: `[M, N]` (weighted sum over k experts). When absent: `[M, k, N]` (per-expert results). |

### 4.4 Type Constraints

| Constraint | Types |
|---|---|
| `T` | `tensor(float)`, `tensor(float16)`, `tensor(bfloat16)` |

`group_indices` is always `tensor(int64)`.

### 4.5 Attributes

None. All configuration is expressed through inputs (following ONNX's general preference for
inputs over attributes when the values may be dynamic).

### 4.6 Shape Inference Rules

Let:
- `M  = input.shape[0]`
- `K  = input.shape[1]`
- `G  = weights.shape[0]`
- `N  = weights.shape[2]`
- `k  = group_indices.shape[1]`

Validation checks (raise error if violated):
1. `input.rank == 2`
2. `weights.rank == 3`
3. `group_indices.rank == 2` and `group_indices.shape[0] == M`
4. `weights.shape[1] == K` (contraction dimension agrees)
5. If `combine_weights` present: `combine_weights.shape == [M, k]`
6. If `bias` present: `bias.shape == [G, N]`

Output shape:
```
output.shape = [M, N]       if combine_weights is present
             = [M, k, N]    otherwise
```

---

## 5. Semantics

### 5.1 Pseudocode

```python
# Inputs:
#   input:           [M, K]
#   weights:         [G, K, N]
#   group_indices:   [M, k]           integers in [0, G)
#   combine_weights: [M, k] or None
#   bias:            [G, N]  or None
#
# r[i, j] = input[i] @ weights[group_indices[i, j]]  (+ bias if present)

r = np.zeros((M, k, N), dtype=input.dtype)

for i in range(M):
    for j in range(k):
        g = group_indices[i, j]
        assert 0 <= g < G, "group index out of range"
        r[i, j] = input[i] @ weights[g]    # [K] @ [K, N] -> [N]
        if bias is not None:
            r[i, j] += bias[g]

if combine_weights is not None:
    # Weighted sum over the k expert slots -> [M, N]
    output = np.einsum('ijk,ij->ik', r, combine_weights)
else:
    # Return per-expert results -> [M, k, N]
    output = r
```

### 5.2 Edge Cases

| Case | Behaviour |
|---|---|
| Empty group (`g` receives no selections) | `weights[g]` is unused; valid. |
| `k == 1` without combine | Dense grouped matmul: one expert per token, output `[M, 1, N]`. |
| `k == 1` with combine | Effectively scales each token result by `combine_weights[i, 0]`. |
| `G == 1`, all indices 0 | Equivalent to `MatMul(input, weights[0])` (+ optional bias). |
| `M == 0` | Zero-token input; output shape is `[0, N]` or `[0, k, N]`; no compute required. |
| Out-of-range index | Implementation-defined error (must not silently produce garbage). Implementors should raise an error or produce undefined results only for out-of-range indices. |

---

## 6. Reference Implementation (Python)

This implementation is intended to be placed at
`onnx/reference/ops/op_grouped_matmul.py` in the onnx/onnx repository.

```python
# SPDX-License-Identifier: Apache-2.0
"""Reference implementation for the GroupedMatMul operator."""

import numpy as np
from onnx.reference.op_run import OpRun


class GroupedMatMul(OpRun):
    """
    GroupedMatMul(input, weights, group_indices[, combine_weights[, bias]])

    Computes a grouped (expert) matrix multiplication as used in
    Mixture-of-Experts feed-forward layers.

    For each token i and expert slot j:
        r[i, j] = input[i] @ weights[group_indices[i, j]]
                  (+ bias[group_indices[i, j]] if bias is provided)

    If combine_weights is provided:
        output[i] = sum_j combine_weights[i, j] * r[i, j]    shape [M, N]
    Otherwise:
        output = r                                             shape [M, k, N]
    """

    op_domain = "ai.onnx"

    def _run(self, input, weights, group_indices, combine_weights=None, bias=None):
        M, K = input.shape
        G, wK, N = weights.shape
        assert wK == K, f"weights dim 1 ({wK}) must equal input dim 1 ({K})"
        assert group_indices.shape == (M, int(group_indices.shape[1])), \
            "group_indices must have shape (M, k)"
        k = group_indices.shape[1]

        # Promote to float64 for numerical stability in the reference.
        dtype = input.dtype
        inp = input.astype(np.float64)
        wts = weights.astype(np.float64)

        # Compute per-expert results: r[i, j] = inp[i] @ wts[g] (+ bias)
        r = np.zeros((M, k, N), dtype=np.float64)
        for i in range(M):
            for j in range(k):
                g = int(group_indices[i, j])
                if g < 0 or g >= G:
                    raise ValueError(
                        f"group_indices[{i},{j}] = {g} is out of range [0, {G})"
                    )
                r[i, j] = inp[i] @ wts[g]
                if bias is not None:
                    r[i, j] += bias[g].astype(np.float64)

        if combine_weights is not None:
            # Weighted sum: output[i] = sum_j combine_weights[i, j] * r[i, j]
            cw = combine_weights.astype(np.float64)   # [M, k]
            output = np.einsum("ijk,ij->ik", r, cw)   # [M, N]
        else:
            output = r  # [M, k, N]

        return (output.astype(dtype),)
```

---

## 7. Test Cases

These cases are intended for `onnx/backend/test/case/node/groupedmatmul.py`.

### Test 1 — Dense (k=1), no combine, no bias

```python
# 4 tokens, K=3, G=2 groups, N=2, k=1
input          = [[1, 0, -1],
                  [0, 1,  2],
                  [1, 1,  0],
                  [0, 0,  1]]          # shape [4, 3]

weights        = [[[1, 0], [0, 1], [-1, 0]],
                  [[0, 1], [1, 0], [ 0, 1]]]  # shape [2, 3, 2]

group_indices  = [[0], [1], [0], [1]]  # shape [4, 1]

# Expected output shape [4, 1, 2]:
#   token 0 -> group 0: [1,0,-1] @ [[1,0],[0,1],[-1,0]] = [1+0+1, 0+0+0] = [2, 0]
#   token 1 -> group 1: [0,1, 2] @ [[0,1],[1,0],[ 0,1]] = [0+1+0, 0+0+2] = [1, 2]
#   token 2 -> group 0: [1,1, 0] @ [[1,0],[0,1],[-1,0]] = [1+0+0, 0+1+0] = [1, 1]
#   token 3 -> group 1: [0,0, 1] @ [[0,1],[1,0],[ 0,1]] = [0+0+0, 0+0+1] = [0, 1]
output         = [[[2, 0]], [[1, 2]], [[1, 1]], [[0, 1]]]  # shape [4, 1, 2]
```

### Test 2 — Top-k (k=2), no combine, with bias

```python
M, k, K, G, N = 2, 2, 2, 3, 2
input         = [[1.0, 0.0], [0.0, 1.0]]         # [2, 2]
weights       = [[[1,0],[0,1]], [[0,1],[1,0]], [[1,1],[0,0]]]  # [3,2,2]
group_indices = [[0, 1], [2, 0]]                  # [2, 2]
bias          = [[0.1, 0.2], [0.3, 0.0], [0.5, 0.5]]  # [3, 2]

# token 0, slot 0 -> g=0: [1,0] @ [[1,0],[0,1]] + [0.1,0.2] = [1.1, 0.2]
# token 0, slot 1 -> g=1: [1,0] @ [[0,1],[1,0]] + [0.3,0.0] = [0.3, 1.0]
# token 1, slot 0 -> g=2: [0,1] @ [[1,1],[0,0]] + [0.5,0.5] = [0.5, 0.5]
# token 1, slot 1 -> g=0: [0,1] @ [[1,0],[0,1]] + [0.1,0.2] = [0.1, 1.2]
output = [[[1.1, 0.2], [0.3, 1.0]],
          [[0.5, 0.5], [0.1, 1.2]]]   # [2, 2, 2]
```

### Test 3 — Top-k (k=2) with combine, no bias

```python
M, k, K, G, N = 2, 2, 2, 3, 2
# (same input/weights/group_indices as Test 2, no bias)
combine_weights = [[0.6, 0.4], [0.3, 0.7]]       # [2, 2]

# token 0: 0.6*[1,0] + 0.4*[0,1] = [0.6, 0.4]
# token 1: 0.3*[0,1] + 0.7*[0,1] = [0.0, 1.0]
output = [[0.6, 0.4], [0.0, 1.0]]                 # [2, 2]
```

### Test 4 — Empty group (one expert unused)

```python
# Group 1 receives no tokens.
M, k, K, G, N = 4, 1, 2, 3, 2
group_indices = [[0], [0], [2], [2]]   # group 1 unused
# weights[1] is never accessed; output is well-defined.
```

### Test 5 — Single group (degenerates to MatMul)

```python
G = 1
group_indices = [[0], [0], [0]]        # all tokens -> group 0
# output == MatMul(input, weights[0]) (reshaped from [M,1,N] to [M,1,N])
```

---

## 8. Typical Usage — MoE Feed-Forward Layer

A standard top-k MoE FFN with two projections maps directly onto two `GroupedMatMul` ops.
No `Expand` of the token batch is required — the op reuses each token row across its `k`
selected experts internally.

```
# Notation: B = batch, S = sequence length, H = hidden dim, F = FFN inner dim
# E = num_experts, k = experts_per_token

scores          = Softmax(MatMul(hidden, router_W))          # [B, S, E]
values, indices = TopK(scores, k)                            # [B, S, k]

h   = Reshape(hidden,  [B*S, H])
idx = Reshape(indices, [B*S, k])
val = Reshape(values,  [B*S, k])

# --- Up projection: per-expert output, no combine ---
# output shape: [B*S, k, F]
h_up = GroupedMatMul(h, expert_gate_W, idx)       # + expert_gate_bias (optional)
h_up = SiLU(h_up)

# Reshape for down projection: treat each (token, expert-slot) pair as a row.
h_flat  = Reshape(h_up, [B*S*k, F])
idx2    = Reshape(idx,  [B*S*k, 1])               # each flat row still in one expert
val2    = Reshape(val,  [B*S*k, 1])               # combine weight (k=1 -> simple scale)

# --- Down projection: fused weighted-sum combine ---
# output shape: [B*S, H]
out = GroupedMatMul(h_flat, expert_down_W, idx2, val2)
out = Reshape(out, [B, S, H])
```

The up-projection uses the **no-combine** form because the activation must run per expert
before the reduce. The down-projection uses the **combine** form to fuse the top-k weighted
sum, avoiding a separate `Mul + ReduceSum` and the round-trip of the `[B*S*k, H]`
intermediate through memory.

---

## 9. Design Alternatives Considered

### 9.1 Single `[M, k]` index tensor vs. separate 2-D/3-D variants

The current proposal uses a single `[M, k]` `group_indices` tensor, with `k=1` as the dense
case. An alternative would be separate op signatures for the dense and top-k cases.

**Decision: single `[M, k]` form.** This keeps the spec and every kernel to one code path.
Callers flatten any leading batch dimensions into `M` first; in ONNX Runtime a `Reshape` of
this kind is a zero-copy metadata-only operation.

### 9.2 Fused activation

An earlier draft included an optional `activation` attribute (`"none"`, `"silu"`, `"relu"`).

**Decision: no fused activation.** Activations stay as separate ONNX ops to keep the graph
composable across the many MoE routing variants. The combine step is fused because it is
common to all variants and cannot otherwise avoid the `Expand`/`ReduceSum` memory overhead.

### 9.3 `group_indices` vs. `group_offsets`

An alternative interface uses a sorted token buffer and integer offsets instead of unsorted
indices. This matches the cuBLAS grouped-GEMM API more directly.

**Decision: `group_indices` (unsorted).** Indices compose naturally with `TopK`/`Gather`
and do not require callers to pre-sort the token batch. Runtimes sort internally (both the
CPU and CUDA implementations do so). This keeps the op declarative, matching ONNX's role as
a model interchange format rather than a kernel description language.

### 9.4 Stacked 3-D weights `[G, K, N]` vs. a list of variable-size matrices

An alternative would be a sequence of weight tensors with potentially different `K`/`N`
dimensions per expert (the general "heterogeneous-expert" case).

**Decision: stacked `[G, K, N]`.** All experts sharing `K` and `N` is the overwhelmingly
common case in deployed MoE models. Homogeneous shapes keep memory contiguous, enable
static shape inference, and simplify every implementation.

### 9.5 Batched leading dimensions in `input`

An alternative would accept `input` of shape `[..., K]` with an arbitrary leading batch.

**Decision: require `input` to be 2-D `[M, K]`.** Callers flatten leading dims via
`Reshape`, which is zero-copy in ONNX Runtime. This keeps the spec and every implementation
to a single flat-indexing code path without losing any expressiveness.

---

## 10. Relationship to Existing ONNX Operators

| Existing op | Relationship |
|---|---|
| `MatMul` | `GroupedMatMul` with `G=1`, all indices 0, no combine is identical to `MatMul(input, weights[0])`. |
| `GatherND` + `MatMul` | The reference decomposition (see §3) uses these; `GroupedMatMul` is a fused efficient version. |
| `Einsum` | Cannot express the dynamic routing (index-dependent contraction) without materialising the gathered weight slice. |
| `QLinearMatMul` | Quantised MatMul; a future `QLinearGroupedMatMul` could follow the same design. |

---

## 11. Files Required for onnx/onnx PR

The following files must be added or modified in the `onnx/onnx` repository to complete the
standardisation, as described in
[`docs/AddNewOp.md`](https://github.com/onnx/onnx/blob/main/docs/AddNewOp.md):

| Component | File |
|---|---|
| Schema definition (C++) | `onnx/defs/math/defs.cc` — add `ONNX_OPERATOR_SET_SCHEMA(GroupedMatMul, <opset>, ...)` |
| Operator set registration | `onnx/defs/operator_sets.h` — add entry in current opset block |
| Type/shape inference | Inline in schema via `.TypeAndShapeInferenceFunction(...)` (see §4.6) |
| Python reference implementation | `onnx/reference/ops/op_grouped_matmul.py` (see §6) |
| Node tests | `onnx/backend/test/case/node/groupedmatmul.py` (see §7) |
| Shape inference tests | `onnx/test/shape_inference_test.py` |
| Upgrade tests | `onnx/test/version_converter/automatic_upgrade_test.py` |
| Downgrade tests | `onnx/test/version_converter/automatic_downgrade_test.py` |

The reference ONNX Runtime implementation (`com.microsoft.GroupedMatMul`) provides CPU and
CUDA kernels that can guide implementors:

| File | Description |
|---|---|
| `onnxruntime/contrib_ops/cpu/grouped_matmul.cc` | CPU kernel (gather + MLAS GEMM per group + scatter/combine) |
| `onnxruntime/contrib_ops/cuda/grouped_matmul.cc` | CUDA kernel (sorted gather + cuBLAS GEMM per group + scatter/combine) |
| `onnxruntime/contrib_ops/cuda/grouped_matmul_impl.cu` | CUDA gather/scatter/combine device kernels |
| `onnxruntime/test/contrib_ops/grouped_matmul_test.cc` | Tests covering dense, top-k, combine, bias, empty-group, and float16 paths |
| `docs/GroupedMatMul.md` | Full design specification with MoE usage diagram |
