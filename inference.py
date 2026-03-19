from model import HMERModel, greedy_decode, beam_search_decode, DEVICE
from PIL import Image
import torch

def load_model(checkpoint_path: str) -> HMERModel:
    model = HMERModel().to(DEVICE)
    ckpt  = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

if __name__ == "__main__":
    model = load_model("hmer_best.pt")
    img   = Image.open("sample.png")

    print("Greedy :", greedy_decode(model, img))
    print("Beam   :", beam_search_decode(model, img, beam_size=5))
