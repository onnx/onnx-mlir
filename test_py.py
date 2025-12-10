import numpy as np  
from PyRuntime import OMExecutionSession  
  
# 모델 로드  
session = OMExecutionSession('test_matmul.so')  
  
# 입력 데이터 준비  
input_data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  
  
# 실행  
outputs = session.run([input_data])  
  
# 결과 출력  
print("출력:", outputs[0])
