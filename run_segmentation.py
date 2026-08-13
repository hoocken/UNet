from model.unet import UNet
from omegaconf import DictConfig
import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


def load_unet(weights: str) -> UNet:
    checkpoint = torch.load(weights, weights_only=True)
    state_dict = checkpoint['unet_state_dict']

    num_classes = state_dict['out_conv.weight'].shape[0]
    down_convs = [ down.split('.')[1] for down in state_dict if 'down_convs' in down ]
    num_layers = int(down_convs[-1]) + 2

    deepest_conv = [ conv for conv in state_dict if 'conv.conv' in conv and 'weight' in conv ]
    max_channels = state_dict[deepest_conv[-1]].shape[0]

    unet = UNet(num_classes, num_layers, max_channels, 0)

    # Remove deep supervision heads (if available)
    module_prefix = 'deep_supervision_heads'
    keys_to_remove = [k for k in state_dict.keys() if k.startswith(module_prefix)]

    for k in keys_to_remove:
        del state_dict[k]

    unet.load_state_dict(state_dict)

    return unet

"""
Mapping:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] # spine, scapula left, scapula right, sternum, clavicle left, clavicle right, lung_lower_lobe_left, lung_upper_lobe_left, lung_lower_lobe_right, lung_middle_lobe_right, lung_upper_lobe_right, heart
"""
@hydra.main(version_base=None, config_path='config', config_name='config_run')
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lung_heart_unet = load_unet(config.lung_heart_weights).to(device)
    bones_unet = load_unet(config.bones_weights).to(device)
    
    image = Image.open(config.input_image).convert('L')
    image = pil_to_tensor(image).to(dtype=torch.float32, device=device) / 255
    image = image.unsqueeze(0).to(device)

    if config.label_image:
        # TODO: Add measurement of dice scores (for evaluation)
        label = np.load(config.label_image)

    lung_heart_result = lung_heart_unet(image)
    bones_result = bones_unet(image)

    result = torch.cat([bones_result[0][1:], lung_heart_result[0][1:]]) # Concat the channels

    np.save(config.output_folder + '/segmentation_result.npy', result.detach().cpu())
    # plt.imsave("result_plot.png", result.cpu()[13] > 0.5)
    # plt.imsave("label_plot.png", label[18])

if __name__ == "__main__":
    main()