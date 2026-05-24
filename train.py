from pathlib import Path

from omegaconf import DictConfig
import yaml
import hydra

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
        limit_files=40,
    )

    solver = Solver(
        train_ld,
        valid_ld,
        config.train
    )

    solver.train()


if __name__ == "__main__":
    main()