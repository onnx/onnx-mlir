# ONNX to KRNL work

Please add row in first table when adding a new ONNX operator, and move to second table when work is completed

ONNX operations for which some work is needed. 
* M for multi-broadcast, U for unidir-broadcast


ONNX Oper | Person working on it | ONNX 2 KRNL  | Basic functionality | Extended functionality (e.g. broadcast)
----------|----------------------|--------------|---------------------|----------------------------------------
Add       |  ?                   | v            | v                   | noM
And       |  ?                   | v            | v                   | noM
Cosh      |  ?                   | v            | v                   | noM
Div       |  ?                   | v            | v                   | 
Exp       |  ?                   | v            | v                   | 
FullGemm  |                      |              |                     | noU
Gemm      |                      |              |                     | noU
MatMul    |                      |              |                     | noM
Mul       |  ?                   | v            | v                   | noM
Or        |  ?                   | v            | v                   | noM
Relu      |  ?                   | v            | v                   | 
Sigmoid   |  ?                   | v            | v                   |
Sinh      |  ?                   | v            | v                   |
Sub       |  ?                   | v            | v                   | noM
Tanh      |  ?                   | v            | v                   |
Xor       |  ?                   | v            | v                   | noM


ONNX operations for which the work is completed (full functionality) and tested

ONNX Oper | Person working on it | Initial work | Basic functionality | Extended functionality (e.g. broadcast)
----------|----------------------|--------------|---------------------|----------------------------------------
