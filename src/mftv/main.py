import pprint
import sys

import hydra
from omegaconf import OmegaConf
import numpy as np
import lightning.pytorch as pl
import torch

import conf.conf
import dcon.utils


log = dcon.utils.getLoggerByFilename(__file__)


@hydra.main(**dcon.utils.HYDRA_INIT)
def main(cfg):
    with conf.conf.Session() as db:
        cfg = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv[:-1]))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    rng = np.random.default_rng(dcon.utils.RNG_RANDBITS['DATASET'][cfg.rng_seed])

    logger = dcon.loggers.CSVLogger(cfg.run_dir, name=None)

    callbacks = []
    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_epochs=10,
        reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    if cfg.fit:
        pass
        # trainer.fit(model, datamodule=dataset)


if __name__ == '__main__':
    last_override, run_dir = conf.conf.get_run_dir()
    conf.conf.set_run_dir(last_override, run_dir)
    main()
