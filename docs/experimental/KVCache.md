<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Experimental KV-cache type (fork only)

> **This is a coordinated ONNX fork experiment, not an adopted ONNX standard
> feature.** The IR and operator versions below are fork-local prototypes and
> must not be treated as available in released ONNX packages.

This experiment adds the dedicated `TypeProto.kv_cache_type` at protobuf field
10 and uses fork IR version 15 and `ai.onnx` opset 29. The cache is a
runtime-owned value. The type contains only logical payload dtype, batch size,
KV-head count, separate key and value head widths, and the sliding-window
contract. It does not describe pages, rings, allocation addresses, devices,
quantization encodings, a C++ object layout, or serialized runtime state.

Use `onnx.helper.make_kv_cache_type_proto` or
`onnx.helper.make_kv_cache_type` to construct the type. Each logical dimension
may be an integer, a symbolic name, or unknown. The optional `sliding_window`
argument is an integer: zero means full history; a positive value is the
inclusive causal window width. The canonical C++ type-constraint spelling is
`kv_cache(float)`, `kv_cache(float16)`, or `kv_cache(bfloat16)`; those strings
identify payload dtype only. Use the protobuf helper to preserve geometry and
window metadata. The ONNX text format accepts
`kv_cache(float)[B,Hkv,Dk,Dv];W`.

## Operators

`KVCacheCreate() -> cache` constructs an empty value. It has the required
`elem_type` attribute and optional `batch_size`, `num_kv_heads`,
`key_head_size`, and `value_head_size` attributes. `sliding_window` defaults
to zero. The inferred output type records these attributes. It never embeds a
runtime payload in a model initializer.

`AttentionWithCache(Q, K_new, V_new, cache, attn_mask?, positions?) ->
(Y, cache_next)` uses BNSD tensors. It appends new keys and values logically,
attends over the resulting history, and returns the output plus a new cache
value version. Storage may be reused only when no observable input version
can be mutated; otherwise implementations must use copy-on-write or distinct
storage. The cache value's `sliding_window` and the operation's attribute
must match. A positive window requires `is_causal=1`. For absolute query
position `q` and key position `p`, visible keys satisfy `p <= q` and, for
window width `W > 0`, `p >= q-W+1`. Thus W=1 includes the current token only.
Chunked prefill applies this bound separately to each query while attending
over the old cache and all newly appended keys. After the append, a sliding
cache retains the final half-open logical range
`[max(0, end-W), end)`. Physical ring/page choice is runtime-only.

`KVCacheExport(cache) -> (K, V, positions)` materializes retained logical
entries in increasing position order. It is a diagnostic/interoperability
operation and may require gathering or transfer. `KVCacheDescribe(cache) ->
(lengths, retained_range)` returns computed end positions and per-row
half-open retained intervals; it does not export payload.

## Initial limits

The initial operator contract is 4-D BNSD self-attention with equal query,
new-key, and new-value sequence lengths, uniform batch end positions, and
contiguous appends. The optional `positions` input is int64 `[B, Sk]` and, if
present, must provide those next contiguous absolute positions. GQA is
supported when the query head count is divisible by the KV-head count, using
the standard contiguous query-head grouping. Key and value widths may differ.
The initial contract does not cover ragged per-row appends, position gaps,
noncausal sliding windows, cross-attention, mixed/quantized payload types,
score diagnostics, or mutable aliasing.

Attention uses the established ONNX `Attention` mask, scale, head mapping,
softcap, and softmax-precision semantics. Masks address logical key positions,
not physical storage slots. Implementations must reject combinations they
cannot support rather than silently changing visibility or cache state.

An independent runtime/model converter must use the protobuf helpers and
generated TypeProto fields; support in `onnx_ir` is not implied by this fork.
