import torch  
import torch.onnx  
import models
from models.experimental import attempt_load 
# 假设input_data是模型的输入数据，它应该是一个torch.Tensor或者一个包含torch.Tensor的列表/元组  
# 'model' 是你的PyTorch模型  
# 'onnx_filename' 是你想要保存的ONNX模型的文件名  

model=attempt_load('/home/mjy/yolov5/runs/train/exp2/weights/best.pt')  
input_data= torch.zeros(1, 4, 640,640)
torch.onnx.export(model,               # model being run  
                  input_data,           # model input (or a tuple for multiple inputs)  
                  "onnx_filename.onnx",  # where to save the model (can be a file or file-like object)  
                  export_params=True,    # store the trained parameter weights inside the model file  
                  opset_version=12,      # the ONNX version to export the model to  
                  do_constant_folding=True, # whether to execute constant folding for optimization  
                  input_names = ['input'],   # the model's input names  
                  output_names = ['output'], # the model's output names  
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes  
                               'output' : {0 : 'batch_size'}})