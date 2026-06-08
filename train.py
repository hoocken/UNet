from pathlib import Path

from omegaconf import DictConfig
import yaml
import hydra
import matplotlib.pyplot as plt
import numpy as np

from data.dataset import build_loader
from solver import Solver


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config: DictConfig):
    config_data = config.data
    config_train = config.train
    
    train_ld, valid_ld = build_loader(
        config_data.filepath,
        config_data.image_dir,
        config_data.labels_dir,
        batch_size=config_data.batch_size,
        # limit_files=50,
    )

    solver = Solver(
        train_ld,
        valid_ld,
        config.train
    )

    solver.train()


if __name__ == "__main__":
    main()
    # labels_path = "/media/data/student/paxraypp/labels_unpacked/labels_converted/"
    # name = "RSNAPE_56314d0397f4_ba8be36234c2_lateral.npy"
    # arr = np.load(labels_path + name)
    # print(arr.shape)
    # plt.imsave("label_plot.png", arr[4])

    # 3. Save as image
    # plt.savefig("label_plot.png", dpi=300, bbox_inches="tight")