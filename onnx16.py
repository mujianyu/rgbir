
import onnx  
from onnx import numpy_helper  
import numpy  
# 加载现有的 float32 ONNX 模型  
model = onnx.load("/home/mjy/yolov5/runs/train/exp30/weights/best.onnx")  
  
# 遍历模型的初始化器（权重和偏置）  
for initializer in model.graph.initializer:  
    # 将权重从 float32 转换为 float16  
    if initializer.data_type == onnx.TensorProto.FLOAT:  
        data = numpy_helper.to_array(initializer)  
        data = data.astype(numpy.float16)  
          
        # 更新初始化器的数据  
        initializer.data_type = onnx.TensorProto.FLOAT16  
        initializer.raw_data = data.tobytes()  
  
# 保存新的 float16 ONNX 模型  
onnx.save(model, "/home/mjy/yolov5/runs/train/exp30/weights/best.onnx")