import logging
import hydra
import wandb
from src import train

logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb.login()
    train(cfg, wandb, logging)


if __name__ == "__main__":
    main()
