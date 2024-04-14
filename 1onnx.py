import onnx
from onnxruntime.transformers.onnx_model import OnnxModel

from tqdm import tqdm

def has_same_value(val_one,val_two):
    if val_one.raw_data == val_two.raw_data:
        return True
    else:
        return False

path = f"/home/mjy/yolov5/runs/train/exp10/weights/best.onnx"  # 242M
output_path = f"/home/mjy/yolov5/runs/train/exp10/weights/best.onnx"  # 7.50M
model = onnx.load(path)
onnx_model = OnnxModel(model)

count = len(model.graph.initializer)
same = [-1] * count
for i in tqdm(range(count - 1)):
  if same[i] >= 0:
        continue
  for j in range(i+1, count):
      if has_same_value(model.graph.initializer[i], 
                        model.graph.initializer[j]):
          same[j] = i

for i in tqdm(range(count)):
   if same[i] >= 0:
       onnx_model.replace_input_of_all_nodes(model.graph.initializer[i].name,
                                             model.graph.initializer[same[i]].name)
onnx_model.update_graph()
onnx_model.save_model_to_file(output_path)
