from omegaconf import OmegaConf


def train(cfg, wandb, logger):
    logger.info(OmegaConf.to_yaml(cfg))
    if wandb is None:
        logger.warning("wandb is None!!!!!!!!")

