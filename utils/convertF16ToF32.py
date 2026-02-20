import onnx
import numpy as np
from onnx import numpy_helper, external_data_helper


def convert_fp16_to_fp32_with_external_data(
    input_model_path: str, output_model_path: str
):
    # Load the model (with external weights)
    model = onnx.load_model(
        input_model_path,
        load_external_data=True,  # Necessary to read the external tensors
    )

    # Convert all initializers (weights)
    for tensor in model.graph.initializer:
        if tensor.data_type == onnx.TensorProto.FLOAT16:
            array_fp16 = numpy_helper.to_array(tensor)
            array_fp32 = array_fp16.astype(np.float32)
            tensor_fp32 = numpy_helper.from_array(array_fp32, tensor.name)
            tensor.CopyFrom(tensor_fp32)

    # Update type in inputs/outputs/value_info
    def update_elem_type_to_fp32(value_info):
        if value_info.type.HasField("tensor_type"):
            tt = value_info.type.tensor_type
            if tt.elem_type == onnx.TensorProto.FLOAT16:
                tt.elem_type = onnx.TensorProto.FLOAT

    for vi in model.graph.input:
        update_elem_type_to_fp32(vi)
    for vi in model.graph.output:
        update_elem_type_to_fp32(vi)
    for vi in model.graph.value_info:
        update_elem_type_to_fp32(vi)

    # Update node attributes with FP16 tensors
    for node in model.graph.node:
        for attr in node.attribute:
            if (
                attr.type == onnx.AttributeProto.TENSOR
                and attr.t.data_type == onnx.TensorProto.FLOAT16
            ):
                arr_fp16 = numpy_helper.to_array(attr.t)
                arr_fp32 = arr_fp16.astype(np.float32)
                attr_fp32 = numpy_helper.from_array(arr_fp32)
                attr.t.CopyFrom(attr_fp32)

    # Save model with external data
    onnx.save_model(
        model,
        output_model_path,
        save_as_external_data=True,  # Keep external data format
        all_tensors_to_one_file=True,  # Or False if you prefer individual files
        location="model_fp32.data",  # Change this if needed
        size_threshold=1024,  # Keep small weights inline
        convert_attribute=True,
    )


# Example usage:

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True, help="Path of the input onnx model(float16)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path of the output onnx model(converted to float32",
    )
    args = parser.parse_args()
    print(args.input, args.output)
    convert_fp16_to_fp32_with_external_data(args.input, args.output)
