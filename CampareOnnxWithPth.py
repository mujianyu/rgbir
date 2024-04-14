# 测试onnx与torch的输出 
#使用随机输入input1进行测试,看torch下的模型和onnx模型输出是否一致.判断方法为采用np.testing.assert_almost_equal进行测试,判断输出的小数点后三位,如果一致,输出结果为None.
import onnxruntime
import torch
from models.experimental import attempt_load
import numpy as np
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
 
onnx_weights = '/home/mjy/yolov5/runs/train/exp2/weights/best.onnx'  # onnx权重路径
torch_weights = '/home/mjy/yolov5/runs/train/exp2/weights/best.pt'  # torch权重路径
session = onnxruntime.InferenceSession(onnx_weights)
model = attempt_load(torch_weights)
 
print(session)
input1 = torch.randn(1, 4, 640, 640)   # tensor
img = input1.numpy().astype(np.float32)  # array
model.eval()
with torch.no_grad():
    torch_output = model(input1)[0]

input_name = session.get_inputs()[0].name  
output_name = session.get_outputs()[0].name  
  
# 运行模型  
#output = session.run([output_name], {input_name: img})
#print(input_name)
#print(output_name)

onnx_output = torch.tensor(session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: img}))
#判断输出结果是否一致，小数点后3位一致即可
print("onnx的输出")
print(onnx_output[0])
print("torch的输出")
print(to_numpy(torch_output))
print(np.testing.assert_almost_equal(to_numpy(torch_output), onnx_output[0], decimal=3))

# provider = "CPUExecutionProvider"
# onnx_session = onnxruntime.InferenceSession(onnx_weights, providers=[provider])

# print("----------------- 输入部分 -----------------")
# input_tensors = onnx_session.get_inputs()  # 该 API 会返回列表
# for input_tensor in input_tensors:         # 因为可能有多个输入，所以为列表
    
#     input_info = {
#         "name" : input_tensor.name,
#         "type" : input_tensor.type,
#         "shape": input_tensor.shape,
#     }
#     print(input_info)

# print("----------------- 输出部分 -----------------")
# output_tensors = onnx_session.get_outputs() # 该 API 会返回列表
# for output_tensor in output_tensors:         # 因为可能有多个输出，所以为列表
    
#     output_info = {
#         "name" : output_tensor.name,
#         "type" : output_tensor.type,
#         "shape": output_tensor.shape,
#     }
#     print(output_info)


# print(torch_output.shape)