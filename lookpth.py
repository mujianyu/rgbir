import torch  # 命令行是逐行立即执行的
import torch.nn as nn

content = torch.load('/home/mjy/yolov5/runs/train/exp30/weights/best.pt')
# print(content.keys())   # keys()
# # 之后有其他需求比如要看 key 为 model 的内容有啥
# print(content['model'])
param = content['model']

# for name, module in param.named_modules():  
#         print(name)
# 获取模型的状态字典  
state_dict = param.state_dict()  
# print(state_dict)
# 假设你知道卷积层的权重和偏置的键名  
conv_weight = state_dict['model.0.conv.weight']  # 替换为实际的键名  




import onnx 
model= onnx.load('/home/mjy/yolov5/runs/train/exp30/weights/best.onnx')
for initializer in model.graph.initializer:  
    # 获取权重名  
    name = initializer.name  
    # 获取权重的数据类型  
    data_type = initializer.data_type  
    # 获取权重的值（可能是一个NumPy数组）  
    value = onnx.numpy_helper.to_array(initializer)  
    #float 32
    print(f"Name: {name}, Type: {data_type}, Shape: {value.shape}")