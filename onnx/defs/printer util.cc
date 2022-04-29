/*
 * SPDX-License-Identifier: Apache-2.0
 */

// #include "onnx/defs/printer.h"
#include <iostream>

#include "onnx/onnx_pb.h"

#include "onnx/defs/parser.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

class ProtoPrinter {
 public:
  ProtoPrinter(std::ostream& os) : output(os) {}

  void print(const TensorShapeProto_Dimension& dim);

  void print(const TensorShapeProto& shape);

  void print(const TypeProto_Tensor& tensortype);

  void print(const TypeProto& type);

  void print(const TensorProto& tensor);

  void print(const ValueInfoProto& value_info);

  void print(const ValueInfoList& vilist);

  void print(const AttributeProto& attr);

  void print(const AttrList& attrlist);

  void print(const NodeProto& node);

  void print(const NodeList& nodelist);

  void print(const GraphProto& graph);

  void print(const FunctionProto& fn);

  void print(const OperatorSetIdProto& opset); 

 private:
  template <typename T>
  inline void print(T prim) {
    output << prim;
  }
  
  template <typename Collection>
  inline void print(const char* open, const char* separator, const char* close, Collection coll) {
    const char* sep = "";
    output << open;
    for (auto& elt : coll) {
      output << sep;
      print(elt);
      sep = separator;
    }
    output << close;
  }

  std::ostream& output;
};

void ProtoPrinter::print(const TensorShapeProto_Dimension& dim) {
  if (dim.has_dim_value())
    output << dim.dim_value();
  else if (dim.has_dim_param())
    output << dim.dim_param();
  else
    output << "?";
}

void ProtoPrinter::print(const TensorShapeProto& shape) {
  print("[", ",", "]", shape.dim());
}

void ProtoPrinter::print(const TypeProto_Tensor& tensortype) {
  output << PrimitiveTypeNameMap::ToString(tensortype.elem_type());
  if (tensortype.has_shape()) {
    if (tensortype.shape().dim_size() > 0)
      print(tensortype.shape());
  } else
    output << "[...]";
}

void ProtoPrinter::print(const TypeProto& type) {
  if (type.has_tensor_type())
    print(type.tensor_type());
}

void ProtoPrinter::print(const TensorProto& tensor) {
  output << PrimitiveTypeNameMap::ToString(tensor.data_type());
  print("[", ",", "]", tensor.dims());

  // TODO: does not yet handle raw_data or FLOAT16 or externally stored data.
  // TODO: does not yet handle name of tensor.
  switch (static_cast<TensorProto::DataType>(tensor.data_type())) {
    case TensorProto::DataType::TensorProto_DataType_INT8:
    case TensorProto::DataType::TensorProto_DataType_INT16:
    case TensorProto::DataType::TensorProto_DataType_INT32:
    case TensorProto::DataType::TensorProto_DataType_UINT8:
    case TensorProto::DataType::TensorProto_DataType_UINT16:
    case TensorProto::DataType::TensorProto_DataType_BOOL:
      print(" {", ",", "}", tensor.int32_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_INT64:
      print(" {", ",", "}", tensor.int64_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_UINT32:
    case TensorProto::DataType::TensorProto_DataType_UINT64:
      print(" {", ",", "}", tensor.uint64_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_FLOAT:
      print(" {", ",", "}", tensor.float_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_DOUBLE:
      print(" {", ",", "}", tensor.double_data());
      break;
    case TensorProto::DataType::TensorProto_DataType_STRING: {
      const char* sep = "{";
      for (auto& elt : tensor.string_data()) {
        output << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      output << "}";
      break;
    }
    default:
      break;
  }
}

void ProtoPrinter::print(const ValueInfoProto& value_info) {
  print(value_info.type());
  output << " " << value_info.name();
}

void ProtoPrinter::print(const ValueInfoList& vilist) {
  print("(", ", ", ")", vilist);
}

void ProtoPrinter::print(const AttributeProto& attr) {
  output << attr.name() << " = ";
  switch (attr.type()) {
    case AttributeProto_AttributeType_INT:
      output << attr.i();
      break;
    case AttributeProto_AttributeType_INTS:
      print("[", ", ", "]", attr.ints());
      break;
    case AttributeProto_AttributeType_FLOAT:
      output << attr.f();
      break;
    case AttributeProto_AttributeType_FLOATS:
      print("[", ", ", "]", attr.floats());
      break;
    case AttributeProto_AttributeType_STRING:
      output << "\"" << attr.s() << "\"";
      break;
    case AttributeProto_AttributeType_STRINGS: {
      const char* sep = "[";
      for (auto& elt : attr.strings()) {
        output << sep << "\"" << elt << "\"";
        sep = ", ";
      }
      output << "]";
      break;
    }
    case AttributeProto_AttributeType_GRAPH:
      print(attr.g());
      break;
    default:
      break;
  }
}

void ProtoPrinter::print(const AttrList& attrlist) {
  print("<", ", ", ">", attrlist);
}

void ProtoPrinter::print(const NodeProto& node) {
  print("", ", ", "", node.output());
  output << " = " << node.op_type();
  if (node.attribute_size() > 0)
    print(node.attribute());
  print("(", ", ", ")", node.input());
}

void ProtoPrinter::print(const NodeList& nodelist) {
  print("{\n", "\n", "\n}\n", nodelist);
}

void ProtoPrinter::print(const GraphProto& graph) {
  output << graph.name() << " ";
  print(graph.input());
  output << " => ";
  print(graph.output());
  output << " ";
  print(graph.node());
}

void ProtoPrinter::print(const OperatorSetIdProto& opset) {
  output << "\"" << opset.domain() << "\" : " << opset.version();
}

void ProtoPrinter::print(const FunctionProto& fn) {
  output << "<\n";
  output << "  "
         << "domain: \"" << fn.domain() << "\",\n";
  output << "  "
         << "opset_import: ";
  print("[", ",", "]", fn.opset_import());
  output << "\n>\n";
  output << fn.name() << " ";
  if (fn.attribute_size() > 0)
    print("<", ",", ">", fn.attribute());
  print("(", ", ", ")", fn.input());
  output << " => ";
  print("(", ", ", ")", fn.output());
  output << "\n";
  print(fn.node());
}

} // namespace ONNX_NAMESPACE