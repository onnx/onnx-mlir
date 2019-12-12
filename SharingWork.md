# ONNX to KRNL work

Please add row in first table when adding a new ONNX operator, and move to second table when work is completed

ONNX operations for which some work is needed. 
* M for multi-broadcast, U for unidir-broadcast


| ONNX Oper   | Person working on it  | ONNX 2 KRNL    | Basic functionality   | Extended functionality (e.g. broadcast)  |
| ----------  | --------------------- | -------------- | --------------------- | ---------------------------------------- |
| Add         | Tung (updated)        | v              | v                     | noM                                      |
| And         | Tung                  | v              | v                     | noM                                      |
| Cosh        | Tung                  | v              | v                     | noM                                      |
| Div         | Tung                  | v              | v                     |                                          |
| Elu         | Tung                  | v              | v                     |                                          |
| Exp         | Tung                  | v              | v                     |                                          |
| FullGemm    |                       |                |                       | noU                                      |
| Gemm        |                       |                |                       | noU                                      |
| HardSigmoid | Tung                  | v              | v                     |                                          |
| LeakyRelu   | Tung                  | v              | v                     |                                          |
| MatMul      |                       |                |                       | noM                                      |
| Max         | Tung                  | v              | v                     | noM                                      |
| Min         | Tung                  | v              | v                     | noM                                      |
| Mul         | Tung                  | v              | v                     | noM                                      |
| Or          | Tung                  | v              | v                     | noM                                      |
| Relu        | Tung                  | v              | v                     |                                          |
| Selu        | Tung                  | v              | v                     |                                          |
| Sigmoid     | Tung                  | v              | v                     |                                          |
| Sinh        | Tung                  | v              | v                     |                                          |
| Sub         | Tung                  | v              | v                     | noM                                      |
| Sum         | Tung                  | v              | v                     | noM                                      |
| Tanh        | Tung                  | v              | v                     |                                          |
| Xor         | Tung                  | v              | v                     | noM                                      |


ONNX operations for which the work is completed (full functionality) and tested

| ONNX Oper  | Person working on it   | Initial work   | Basic functionality   | Extended functionality (e.g. broadcast)  |
| ---------- | ---------------------- | -------------- | --------------------- | ---------------------------------------- |
