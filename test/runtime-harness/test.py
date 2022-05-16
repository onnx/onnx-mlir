# Copyright IBM Corporation 2020,2022
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PyRuntime import ExecutionSession
import numpy as np

compiled_model = "/workdir/onnx-mlir/build/roberta-base-11.so"
model_file = "roberta-base-11.onnx"

print("RUN:", compiled_model)

# Create an execution session:
session = ExecutionSession(compiled_model)

input_data = np.array([[0, 713, 822, 16, 98, 205, 2]], dtype=np.int64)

output = session.run([input_data])

print(output)

print(session)
