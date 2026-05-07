import hydra
from omegaconf import DictConfig
from trainer import DenoisingTrainer
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(f'Running Model: {cfg.modelName}')
    print(f'Working directory: {os.getcwd()}')

    # Create trainer - it handles DictConfig automatically now
    trainer = DenoisingTrainer(**cfg)

    if cfg.Train:
        trainer.train()
    elif cfg.Inference:
        trainer.test()
    else:
        trainer.test()


if __name__ == '__main__':
    main()