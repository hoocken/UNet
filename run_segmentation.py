from model.unet import UNet
from omegaconf import DictConfig
import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

MAP_TO_VIBES = {
    0: 69, # vertrabrae
    1: 44, # scapula left
    2: 45, # scapula right
    3: 63, # sternum
    4: 46, # clavicle left
    5: 47, # clavicle right
    6: 11, # lung lower lobe left
    7: 10, # lung upper lobe left
    8: 14, # lung lower lobe right
    9: 13, # lung middle lobe right
    10: 12, # lung upper lobe right
    11: 24, # heart
    12: 25, # ascending aorta -> aorta
    13: 25, # descending aorta -> aorta
    14: 25, # aortic arch -> aorta
}

MAP_TO_CONVERTED_LABELS = {
    0: 1, # vertrabrae
    1: 9, # scapula left
    2: 10, # scapula right
    3: 12, # sternum
    4: 13, # clavicle left
    5: 14, # clavicle right
    6: 2, # lung lower lobe left
    7: 3, # lung upper lobe left
    8: 4, # lung lower lobe right
    9: 5, # lung middle lobe right
    10: 6, # lung upper lobe right
    11: 15, # heart
    12: 16, # ascending aorta -> aorta
    13: 17, # descending aorta -> aorta
    14: 18, # aortic arch -> aorta
}

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

@hydra.main(version_base=None, config_path='config', config_name='config_run')
def main(config: DictConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lung_heart_unet = load_unet(config.lung_heart_weights).to(device)
    bones_unet = load_unet(config.bones_weights).to(device)
    
    image = Image.open(config.input_image).convert('L').resize((512, 512))
    image = pil_to_tensor(image).to(dtype=torch.float32, device=device) / 255
    image = image.unsqueeze(0).to(device)

    if config.label_image:
        # TODO: Add measurement of dice scores (for evaluation)
        label = np.load(config.label_image)

    lung_heart_result = lung_heart_unet(image)
    bones_result = bones_unet(image)

    result = torch.cat([bones_result[0][1:], lung_heart_result[0][1:]]) # Concat the channels

    if config.map_to_vibesegmentator:
        map = MAP_TO_VIBES
        num_channels = 74
    else:
        map = MAP_TO_CONVERTED_LABELS
        num_channels = 19        

    mapped_result = torch.zeros((num_channels, *result.shape[1:]), dtype=torch.bool).to(device)
    for i, v in map.items():
        mapped_result[v] += result[i] > 0.5

    np.save(config.output_folder + '/segmentation_result.npy', mapped_result.detach().cpu())
    plt.imsave("result_plot.png", mapped_result.cpu()[25])
    plt.imsave("label_plot.png", label[18])

if __name__ == "__main__":
    main()