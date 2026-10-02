from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import DesignEmbedder
from data import SareeDataset


def load_model(checkpoint, device=None):
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    model=DesignEmbedder(state.get('embedding_dim',256),pretrained=False)
    model.load_state_dict(state['state_dict']); model.to(device).eval()
    return model,device,state


@torch.inference_mode()
def embed_frame(model, frame, root, device, batch_size=32, workers=0):
    ds=SareeDataset(frame,root,train=False); dl=DataLoader(ds,batch_size=batch_size,num_workers=workers)
    zs=[]
    for x,_,_ in dl: zs.append(model(x.to(device)).cpu().numpy())
    return np.concatenate(zs)


@torch.inference_mode()
def embed_one(model, image_path, device):
    from PIL import Image
    from data import transform
    with Image.open(image_path) as im: x=transform(False)(im.convert('RGB')).unsqueeze(0)
    return model(x.to(device))[0].cpu().numpy()
