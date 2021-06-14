#
# Genrate models for testing. Intended as mostly 'sample' style code that that
# will eventually become part of a broader test system.
#

from pathlib import Path
import tempfile

from assembler import TensorDataSource
from bert import create_bert

import onnx
from onnx import TensorProto


def format_float_mode(float_mode: TensorProto):
  if float_mode == TensorProto.BFLOAT16:
    return "bfloat16"
  elif float_mode == TensorProto.FLOAT:
    return "float32"
  raise "unknown mode"


def test_bert(batch_size: int, layers: int, seqlen: int, heads: int, head_size: int, use_layernorm_op: bool, float_mode: TensorProto):
  model = create_bert(batch_size=batch_size, seqlen=seqlen, layers=layers, heads=heads, head_size=head_size, use_layernorm_op=use_layernorm_op, tensor_data_source=TensorDataSource.MODEL_INPUT, float_mode=float_mode)
  file_name = f"bert.layers_{layers}_batch_{batch_size}_seqlen_{seqlen}_heads_{heads}_headsize_{head_size}{'' if use_layernorm_op else '_layernorm_off'}.{format_float_mode(float_mode)}.onnx"
  file_path = Path.joinpath(Path(tempfile.gettempdir()), Path(file_name))
  onnx.save(model, file_path)
  print(f"Wrote model to {file_path}")


test_bert(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, float_mode=TensorProto.BFLOAT16, use_layernorm_op=True)
test_bert(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, float_mode=TensorProto.FLOAT, use_layernorm_op=True)
test_bert(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, float_mode=TensorProto.BFLOAT16, use_layernorm_op=True)
test_bert(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, float_mode=TensorProto.FLOAT, use_layernorm_op=True)

test_bert(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, float_mode=TensorProto.BFLOAT16, use_layernorm_op=False)
test_bert(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, float_mode=TensorProto.FLOAT, use_layernorm_op=False)
test_bert(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, float_mode=TensorProto.BFLOAT16, use_layernorm_op=False)
test_bert(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, float_mode=TensorProto.FLOAT, use_layernorm_op=False)
