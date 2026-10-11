# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from onnx import helper
from onnx.reference.op_run import OpRun
from onnx.reference.ops.op_attention import _compute_attention


@dataclass(frozen=True)
class _KVCacheValue:
    keys: np.ndarray
    values: np.ndarray
    elem_type: int
    sliding_window: int
    end_position: int


class KVCacheCreate(OpRun):
    def _run(
        self,
        elem_type,
        batch_size=None,
        num_kv_heads=None,
        key_head_size=None,
        value_head_size=None,
        sliding_window=0,
    ):
        dims = (batch_size, num_kv_heads, key_head_size, value_head_size)
        if any(dim is None for dim in dims):
            raise ValueError("The reference KVCacheCreate requires all four logical dimensions.")
        if batch_size < 0 or any(dim <= 0 for dim in dims[1:]):
            raise ValueError("Invalid logical KV-cache dimensions.")
        if sliding_window < 0:
            raise ValueError("sliding_window must be nonnegative.")
        dtype = helper.tensor_dtype_to_np_dtype(elem_type)
        keys = np.empty((batch_size, num_kv_heads, 0, key_head_size), dtype=dtype)
        values = np.empty((batch_size, num_kv_heads, 0, value_head_size), dtype=dtype)
        return (
            _KVCacheValue(
                keys=keys,
                values=values,
                elem_type=elem_type,
                sliding_window=sliding_window,
                end_position=0,
            ),
        )


class AttentionWithCache(OpRun):
    def _run(
        self,
        Q,
        K_new,
        V_new,
        cache,
        attn_mask=None,
        positions=None,
        scale=None,
        is_causal=0,
        q_num_heads=None,
        kv_num_heads=None,
        softmax_precision=None,
        softcap=0.0,
        sliding_window=0,
    ):
        if not isinstance(cache, _KVCacheValue):
            raise TypeError("AttentionWithCache expects a reference KV-cache value.")
        if sliding_window != cache.sliding_window:
            raise ValueError("AttentionWithCache sliding_window must match the cache type.")
        if sliding_window > 0 and is_causal != 1:
            raise ValueError("A sliding-window cache requires is_causal=1.")
        if Q.ndim != 4 or K_new.ndim != 4 or V_new.ndim != 4:
            raise ValueError("AttentionWithCache requires 4-D BNSD inputs.")
        if Q.shape[0] != K_new.shape[0] or Q.shape[0] != V_new.shape[0]:
            raise ValueError("AttentionWithCache batch sizes must match.")
        if Q.shape[2] != K_new.shape[2] or Q.shape[2] != V_new.shape[2]:
            raise ValueError("AttentionWithCache query, key, and value sequence lengths must match.")
        if K_new.shape[2] != V_new.shape[2]:
            raise ValueError("AttentionWithCache K_new and V_new sequence lengths must match.")
        if Q.shape[3] != K_new.shape[3]:
            raise ValueError("AttentionWithCache query and key widths must match.")
        if K_new.shape[1] != V_new.shape[1] or K_new.shape[1] != cache.keys.shape[1]:
            raise ValueError("AttentionWithCache KV head counts must match.")
        if Q.shape[0] != cache.keys.shape[0] or K_new.shape[3] != cache.keys.shape[3]:
            raise ValueError("AttentionWithCache inputs do not match cache geometry.")
        if V_new.shape[3] != cache.values.shape[3]:
            raise ValueError("AttentionWithCache value width does not match cache geometry.")
        if Q.dtype != cache.keys.dtype or K_new.dtype != cache.keys.dtype or V_new.dtype != cache.values.dtype:
            raise ValueError("AttentionWithCache tensor types must match the cache payload type.")
        if Q.shape[1] % K_new.shape[1] != 0:
            raise ValueError("Query head count must be divisible by KV head count.")

        expected_positions = np.arange(
            cache.end_position, cache.end_position + K_new.shape[2], dtype=np.int64
        )
        if positions is not None:
            positions = np.asarray(positions)
            if positions.dtype != np.int64 or positions.shape != (Q.shape[0], K_new.shape[2]):
                raise ValueError("positions must be int64[B,Sk].")
            if not np.all(positions == expected_positions):
                raise ValueError("positions must be contiguous and start at the cache end position.")

        all_keys = np.concatenate((cache.keys, K_new), axis=2)
        all_values = np.concatenate((cache.values, V_new), axis=2)
        output, _, _, _ = _compute_attention(
            Q,
            K_new,
            V_new,
            attn_mask=attn_mask,
            past_key=cache.keys,
            past_value=cache.values,
            scale=scale,
            is_causal=is_causal,
            q_num_heads=q_num_heads,
            kv_num_heads=kv_num_heads,
            softmax_precision=softmax_precision,
            softcap=softcap,
            left_window_size=sliding_window - 1 if sliding_window > 0 else -1,
        )
        end_position = cache.end_position + K_new.shape[2]
        if sliding_window > 0:
            all_keys = all_keys[:, :, -sliding_window:, :]
            all_values = all_values[:, :, -sliding_window:, :]
        next_cache = _KVCacheValue(
            keys=all_keys,
            values=all_values,
            elem_type=cache.elem_type,
            sliding_window=cache.sliding_window,
            end_position=end_position,
        )
        return output, next_cache


class KVCacheExport(OpRun):
    def _run(self, cache):
        if not isinstance(cache, _KVCacheValue):
            raise TypeError("KVCacheExport expects a reference KV-cache value.")
        start_position = cache.end_position - cache.keys.shape[2]
        positions = np.arange(start_position, cache.end_position, dtype=np.int64)
        positions = np.broadcast_to(positions, (cache.keys.shape[0], positions.size)).copy()
        return cache.keys, cache.values, positions


class KVCacheDescribe(OpRun):
    def _run(self, cache):
        if not isinstance(cache, _KVCacheValue):
            raise TypeError("KVCacheDescribe expects a reference KV-cache value.")
        lengths = np.full((cache.keys.shape[0],), cache.end_position, dtype=np.int64)
        start_position = cache.end_position - cache.keys.shape[2]
        retained_range = np.tile(
            np.array([start_position, cache.end_position], dtype=np.int64),
            (cache.keys.shape[0], 1),
        )
        return lengths, retained_range
