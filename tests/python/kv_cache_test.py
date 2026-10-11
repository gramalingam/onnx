# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

import onnx
from onnx import TensorProto, checker, defs, helper, shape_inference
from onnx.reference import ReferenceEvaluator


def _model(graph: onnx.GraphProto) -> onnx.ModelProto:
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 29)],
        ir_version=onnx.IR_VERSION,
    )


def test_kv_cache_schema_registration() -> None:
    assert defs.get_schema("KVCacheCreate", 29).outputs[0].type_str == "TC"
    attention = defs.get_schema("AttentionWithCache", 29)
    assert [formal.name for formal in attention.inputs] == [
        "Q",
        "K_new",
        "V_new",
        "cache",
        "attn_mask",
        "positions",
    ]
    assert [formal.name for formal in attention.outputs] == ["Y", "cache_next"]
    assert defs.get_schema("KVCacheExport", 29).outputs[2].name == "positions"
    assert defs.get_schema("KVCacheDescribe", 29).outputs[1].name == "retained_range"


def test_kv_cache_create_infers_logical_type() -> None:
    node = helper.make_node(
        "KVCacheCreate",
        [],
        ["cache"],
        elem_type=TensorProto.FLOAT,
        batch_size=2,
        num_kv_heads=2,
        key_head_size=4,
        value_head_size=6,
        sliding_window=8,
    )
    graph = helper.make_graph(
        [node],
        "create_cache",
        [],
        [helper.make_empty_tensor_value_info("cache")],
    )
    inferred = shape_inference.infer_shapes(_model(graph))
    cache_type = inferred.graph.output[0].type.kv_cache_type
    assert cache_type.elem_type == TensorProto.FLOAT
    assert cache_type.batch_size.dim_value == 2
    assert cache_type.num_kv_heads.dim_value == 2
    assert cache_type.key_head_size.dim_value == 4
    assert cache_type.value_head_size.dim_value == 6
    assert cache_type.sliding_window == 8
    checker.check_model(inferred)


def test_attention_with_cache_infers_output_and_preserves_cache_type() -> None:
    cache = helper.make_kv_cache_type(
        "cache",
        TensorProto.FLOAT,
        batch_size=2,
        num_kv_heads=2,
        key_head_size=4,
        value_head_size=6,
        sliding_window=3,
    )
    node = helper.make_node(
        "AttentionWithCache",
        ["Q", "K_new", "V_new", "cache"],
        ["Y", "cache_next"],
        is_causal=1,
        sliding_window=3,
    )
    graph = helper.make_graph(
        [node],
        "attention_with_cache",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, [2, 8, 3, 4]),
            helper.make_tensor_value_info("K_new", TensorProto.FLOAT, [2, 2, 3, 4]),
            helper.make_tensor_value_info("V_new", TensorProto.FLOAT, [2, 2, 3, 6]),
            cache,
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
            helper.make_kv_cache_type(
                "cache_next",
                TensorProto.FLOAT,
                batch_size=2,
                num_kv_heads=2,
                key_head_size=4,
                value_head_size=6,
                sliding_window=3,
            ),
        ],
    )
    inferred = shape_inference.infer_shapes(_model(graph))
    assert [
        dim.dim_value for dim in inferred.graph.output[0].type.tensor_type.shape.dim
    ] == [2, 8, 3, 6]
    assert inferred.graph.output[1].type.kv_cache_type.sliding_window == 3
    checker.check_model(inferred)


def test_attention_with_cache_rejects_window_mismatch() -> None:
    cache = helper.make_kv_cache_type("cache", TensorProto.FLOAT, sliding_window=4)
    node = helper.make_node(
        "AttentionWithCache",
        ["Q", "K_new", "V_new", "cache"],
        ["Y", "cache_next"],
        is_causal=1,
        sliding_window=3,
    )
    graph = helper.make_graph(
        [node],
        "attention_window_mismatch",
        [
            helper.make_tensor_value_info("Q", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("K_new", TensorProto.FLOAT, None),
            helper.make_tensor_value_info("V_new", TensorProto.FLOAT, None),
            cache,
        ],
        [
            helper.make_tensor_value_info("Y", TensorProto.FLOAT, None),
            helper.make_empty_tensor_value_info("cache_next"),
        ],
    )
    with pytest.raises(onnx.shape_inference.InferenceError, match="must match the cache type"):
        shape_inference.infer_shapes(_model(graph))


def test_reference_sliding_cache_is_functional_and_chunk_aware() -> None:
    nodes = [
        helper.make_node(
            "KVCacheCreate",
            [],
            ["cache0"],
            elem_type=TensorProto.FLOAT,
            batch_size=1,
            num_kv_heads=1,
            key_head_size=2,
            value_head_size=1,
            sliding_window=2,
        ),
        helper.make_node(
            "AttentionWithCache",
            ["Q0", "K0", "V0", "cache0"],
            ["Y0", "cache1"],
            is_causal=1,
            sliding_window=2,
        ),
        helper.make_node(
            "AttentionWithCache",
            ["Q1", "K1", "V1", "cache1"],
            ["Y1", "cache2"],
            is_causal=1,
            sliding_window=2,
        ),
        helper.make_node("KVCacheExport", ["cache0"], ["K_initial", "V_initial", "P_initial"]),
        helper.make_node("KVCacheExport", ["cache2"], ["K_final", "V_final", "P_final"]),
        helper.make_node("KVCacheDescribe", ["cache2"], ["lengths", "retained_range"]),
    ]
    graph = helper.make_graph(
        nodes,
        "reference_sliding_cache",
        [
            helper.make_tensor_value_info("Q0", TensorProto.FLOAT, [1, 2, 2, 2]),
            helper.make_tensor_value_info("K0", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("V0", TensorProto.FLOAT, [1, 1, 2, 1]),
            helper.make_tensor_value_info("Q1", TensorProto.FLOAT, [1, 2, 1, 2]),
            helper.make_tensor_value_info("K1", TensorProto.FLOAT, [1, 1, 1, 2]),
            helper.make_tensor_value_info("V1", TensorProto.FLOAT, [1, 1, 1, 1]),
        ],
        [
            helper.make_tensor_value_info("Y0", TensorProto.FLOAT, [1, 2, 2, 1]),
            helper.make_tensor_value_info("Y1", TensorProto.FLOAT, [1, 2, 1, 1]),
            helper.make_tensor_value_info("K_initial", TensorProto.FLOAT, [1, 1, 0, 2]),
            helper.make_tensor_value_info("V_initial", TensorProto.FLOAT, [1, 1, 0, 1]),
            helper.make_tensor_value_info("P_initial", TensorProto.INT64, [1, 0]),
            helper.make_tensor_value_info("K_final", TensorProto.FLOAT, [1, 1, 2, 2]),
            helper.make_tensor_value_info("V_final", TensorProto.FLOAT, [1, 1, 2, 1]),
            helper.make_tensor_value_info("P_final", TensorProto.INT64, [1, 2]),
            helper.make_tensor_value_info("lengths", TensorProto.INT64, [1]),
            helper.make_tensor_value_info("retained_range", TensorProto.INT64, [1, 2]),
        ],
    )
    model = _model(graph)
    zero_q = np.zeros((1, 2, 2, 2), dtype=np.float32)
    first_k = np.zeros((1, 1, 2, 2), dtype=np.float32)
    first_v = np.array([[[[1.0], [3.0]]]], dtype=np.float32)
    second_q = np.zeros((1, 2, 1, 2), dtype=np.float32)
    second_k = np.zeros((1, 1, 1, 2), dtype=np.float32)
    second_v = np.array([[[[5.0]]]], dtype=np.float32)
    (
        y0,
        y1,
        k_initial,
        v_initial,
        p_initial,
        _,
        v_final,
        p_final,
        lengths,
        retained_range,
    ) = ReferenceEvaluator(model).run(
        None,
        {
            "Q0": zero_q,
            "K0": first_k,
            "V0": first_v,
            "Q1": second_q,
            "K1": second_k,
            "V1": second_v,
        },
    )
    np.testing.assert_allclose(y0[0, :, :, 0], [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(y1[0, :, 0, 0], [4.0, 4.0])
    assert k_initial.shape == (1, 1, 0, 2)
    assert v_initial.shape == (1, 1, 0, 1)
    assert p_initial.shape == (1, 0)
    np.testing.assert_array_equal(v_final, np.array([[[[3.0], [5.0]]]], dtype=np.float32))
    np.testing.assert_array_equal(p_final, [[1, 2]])
    np.testing.assert_array_equal(lengths, [3])
    np.testing.assert_array_equal(retained_range, [[1, 3]])
