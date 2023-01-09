"# models_weist_repository" 
convert .pt to .ptl
python export.py --weights yolov5s.pt --include torchscript

before in optimizer.py
import torch
from torch.utils.mobile_optimizer import optimize_for_mobile

torchscript_model = "yolov5s.torchscript"
export_model_name = "yolov5s.torchscript.ptl"

model = torch.jit.load(torchscript_model)
optimized_model = optimize_for_mobile(model)
optimized_model._save_for_lite_interpreter(export_model_name)

print(f"mobile optimized model exported to {export_model_name}")