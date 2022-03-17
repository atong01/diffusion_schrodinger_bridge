import torch
import hydra
import os, sys
from omegaconf import DictConfig, OmegaConf
import logging
import submitit
import wandb

sys.path.append("..")


from bridge.runners.ipf import IPFSequential

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig) -> None:
    #wandb.cfg = OmegaConf.to_container(
    #    cfg, resolve=True, throw_on_missing=True
    #)
    #run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project)
    #env = submitit.JobEnvironment()
    log.info(OmegaConf.to_yaml(cfg))
    # log.info(env)
    log.info("Directory: " + os.getcwd())
    ipf = IPFSequential(cfg)
    ipf.train()


if __name__ == "__main__":
    main()
