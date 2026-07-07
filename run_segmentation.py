from data.transform import GaussianNoise
from model.unet import UNet
from omegaconf import DictConfig
import hydra
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor


from data.dataset import build_loader
from solver import Solver
"""
Mapping:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # spine, scapula left, scapula right, sternum, clavicle left, clavicle right, lung_lower_lobe_left, lung_upper_lobe_left, lung_lower_lobe_right, lung_middle_lobe_right, lung_upper_lobe_right, heart
"""
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    config = config.run

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    unet = UNet(config.num_classes, config.num_layers, config.max_channels, 0).to(device)
    checkpoint = torch.load(config.weights, weights_only=True)
    state_dict = checkpoint['unet_state_dict']

    # Remove deep supervision heads (if available)
    module_prefix = 'deep_supervision_heads'
    keys_to_remove = [k for k in state_dict.keys() if k.startswith(module_prefix)]

    for k in keys_to_remove:
        del state_dict[k]

    unet.load_state_dict(state_dict)
    
    # labels_path = "/media/data/student/paxraypp/labels_unpacked/labels_converted/"
    # images_path = "/media/data/student/paxraypp/paxray_images_unfiltered/images_patlas/"
    images_path = "./"
    name = "image_00000copy"
    # name="RibFrac_056_frontal"

    image = Image.open(images_path + name + ".png").convert('L')
    image = pil_to_tensor(image).to(dtype=torch.float32, device=device) / 255

    # pred = np.load(labels_path + name + ".npy")
    transform = GaussianNoise()
    image, _ = transform(image[None, :, :, :], torch.zeros_like(image[None, :, :, :]))
    print(image[0].shape)
    plt.imsave("img_plot.png", image[0, 0].detach().cpu(), cmap='gray')
    # return
    res = unet(image)

    res = res[0, :, :, :].detach().cpu()
    # np.save('segmentation', res)

    plt.imsave("pred_plot.png", res[6] > 0.5)
    # plt.imsave("label_plot.png", pred[14])




if __name__ == "__main__":
    # TODO: Add running from the CLI
    main()