from yolor.helpers import Yolor
from yolor.utils.datasets import LoadStreams, LoadImages
from yolor.utils.download import attempt_download_from_hub
from yolor.models.models import Darknet

import torch_ttnn
import torch
import ttnn

cfg = "../../../yolor/cfg/yolor_p6.cfg"
weights = "kadirnar/yolor-p6"
imgsz = 640
device = "cpu"

weights = attempt_download_from_hub(weights)
model = Darknet(cfg, imgsz)
model.load_state_dict(torch.load(weights, map_location=device)["model"])
model.eval()

img = torch.rand((1, 3, imgsz, imgsz), device=device)  # init img

p = model(img)
device = ttnn.Device = ttnn.open_device(device_id=0)
option = torch_ttnn.TenstorrentBackendOption(device=device)
compiled_model = torch.compile(model, backend="ttnn", options=option)
compiled_model.eval()
compiled_model(img)
