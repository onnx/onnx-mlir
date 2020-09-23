import gen_onnx_mlir as go

onnx_to_mlir_type_dict = { '(': '<[',
    ')': ']>',
    'tensor' : 'TensorOf',
    'seq' : 'SeqOf',
    'map' : 'TupleOf',
    'bool': 'I1',
    #'uint8' : 'AnyI8',
    #uint16' : 'AnyI16',
    #uint32' : 'AnyI32',
    #uint64' : 'AnyI64',
    'uint8' : 'UI8',
    'uint16' : 'UI16',
    'uint32' : 'UI32',
    'uint64' : 'UI64',
    'int8' : 'I8',
    'int16' : 'I16',
    'int32' : 'I32',
    'int64' : 'I64',
    'float16' : 'F16',
    'float' : 'F32',
    'double' : 'F64',
    'unkown' : 'BF16',
    'complex64' : 'Complex<F32>',
    'complex128' : 'Complex<F64>',
    'string' : 'StringType'}

for key,val in onnx_to_mlir_type_dict.items():
    print(key, val, go.parse_type_str(key))