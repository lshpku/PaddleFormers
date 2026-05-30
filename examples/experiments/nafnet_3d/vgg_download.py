import torch
import numpy as np

pth_url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
pth_path = "./vgg19-dcbb9e9d.pth"
npz_path = "./vgg19-dcbb9e9d-converted.npz"

state = torch.load(pth_path, map_location="cpu")

if "state_dict" in state:
    state = state["state_dict"]

np_state = {}
for k, v in state.items():
    if hasattr(v, "detach"):
        np_state[k] = v.detach().cpu().numpy()

np.savez(npz_path, **np_state)
