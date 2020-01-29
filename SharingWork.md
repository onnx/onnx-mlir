# ONNX to KRNL work

Please add row in first table when adding a new ONNX operator, and move to second table when work is completed

ONNX operations for which some work is needed. 
* M for multi-broadcast, U for unidir-broadcast


| ONNX Oper   | Person working on it  | ONNX 2 KRNL    | Basic functionality   | Extended functionality (e.g. broadcast)  |
| ----------  | --------------------- | -------------- | --------------------- | ---------------------------------------- |
| Add         | Tung (updated)        | v              | v                     | M                                        |
| And         | Tung                  | v              | v                     | M                                        |
| Cosh        | Tung                  | v              | v                     |                                          |
| Div         | Tung                  | v              | v                     | M                                        |
| Elu         | Tung                  | v              | v                     |                                          |
| Exp         | Tung                  | v              | v                     |                                          |
| FullGemm    |                       |                |                       | noU                                      |
| Gemm        | Tung                  | v              |                       | U                                        |
| HardSigmoid | Tung                  | v              | v                     |                                          |
| LeakyRelu   | Tung                  | v              | v                     |                                          |
| MatMul      |                       |                |                       | noM                                      |
| Max         | Tung                  | v              | v                     | M                                        |
| Min         | Tung                  | v              | v                     | M                                        |
| Mul         | Tung                  | v              | v                     | M                                        |
| Or          | Tung                  | v              | v                     | M                                        |
| Relu        | Tung                  | v              | v                     |                                          |
| Selu        | Tung                  | v              | v                     |                                          |
| Sigmoid     | Tung                  | v              | v                     |                                          |
| Sinh        | Tung                  | v              | v                     |                                          |
| Softmax     | Tung                  | v              | v                     |                                          |
| Sub         | Tung                  | v              | v                     | M                                        |
| Sum         | Tung                  | v              | v                     | M                                        |
| Tanh        | Tung                  | v              | v                     |                                          |
| Xor         | Tung                  | v              | v                     | M                                        |


ONNX operations for which the work is completed (full functionality) and tested

| ONNX Oper  | Person working on it   | Initial work   | Basic functionality   | Extended functionality (e.g. broadcast)  |
| ---------- | ---------------------- | -------------- | --------------------- | ---------------------------------------- |
