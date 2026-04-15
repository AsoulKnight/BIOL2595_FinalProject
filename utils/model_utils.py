from models.unet_model import UNet
import torch


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    ### change model here ####
    model = UNet(n_channels=3, n_classes=1).to(DEVICE)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    return model