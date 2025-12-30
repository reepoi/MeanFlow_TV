import pprint
import sys

import hydra
from omegaconf import OmegaConf
import numpy as np
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader

import conf.conf
import mftv.callbacks
import mftv.datasets
import mftv.loggers
import mftv.model
import mftv.utils


log = mftv.utils.getLoggerByFilename(__file__)


class MeanFlow(pl.LightningModule):
    def __init__(self, cfg, rng: np.random.Generator, dataset_start, dataset_end, model):
        super().__init__()
        self.cfg = cfg
        self.rng = rng
        self.dataset_start = dataset_start
        self.dataset_end = dataset_end
        self.model = model
        data_end = self.dataset_end.generate(8192)
        self.register_buffer('mean_end', data_end.mean())
        self.register_buffer('std_end', data_end.std())
        self.register_buffer('zero', torch.tensor(0.), persistent=False)
        self.register_buffer('one', torch.tensor(1.), persistent=False)

    def configure_optimizers(self):
        # lr = self.cfg.model.lr
        # optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.model.lr)
        # return dict(
        #     optimizer=optimizer,
        #     lr_scheduler=dict(
        #         scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, total_steps=5_000),
        #     ),
        # )
        return torch.optim.AdamW(self.model.parameters(), lr=self.cfg.model.lr)

    def train_dataloader(self):
        cl = pl.utilities.CombinedLoader(dict(
            start=DataLoader(self.dataset_start, batch_size=self.cfg.model.batch_size),
            end=DataLoader(self.dataset_end, batch_size=self.cfg.model.batch_size),
            _batch=DataLoader(range(100)),
        ), mode='min_size')
        iter(cl)
        return cl

    def training_step(self, batch, batch_idx):
        return self.compute_loss(batch)

    def forward(self, x_start_time, start_time, end_time):
        return self.model(x_start_time, start_time, end_time)

    def compute_loss(self, batch):
        batch['end'] = (batch['end'] - self.mean_end) / self.std_end

        start_time, end_time = torch.tensor(
            self.rng.uniform(size=(2, self.cfg.model.batch_size, 1)),
            dtype=batch['start'].dtype, device=batch['start'].device,
        )

        x_start_time = batch['start'] * start_time + batch['end'] * (1 - start_time)
        # derivative of x_start_time wrt start_time
        velocity = batch['start'] - batch['end']

        dtime = end_time - start_time
        losses = {}
        if self.cfg.model.tv_loss_coeff > 0:
            _, pmean_flow__pstart_time = torch.autograd.functional.jvp(
                self,
                (x_start_time, start_time, end_time),
                v=(self.zero.expand(velocity.shape), self.one.expand(start_time.shape), self.zero.expand(start_time.shape)),
            )
            mean_flow, pmean_flow__px_start_time = torch.autograd.functional.jvp(
                self,
                (x_start_time, start_time, end_time),
                v=(velocity, self.zero.expand(start_time.shape), self.zero.expand(start_time.shape)),
                create_graph=True,
            )
            dmean_flow__dstart_time = pmean_flow__px_start_time.detach() + pmean_flow__pstart_time
            target_mean_flow = velocity + dtime * dmean_flow__dstart_time
            dtime_tv_target = mean_flow.detach() - velocity - dtime * pmean_flow__pstart_time
            dtime_tv_loss = (dtime * pmean_flow__px_start_time - dtime_tv_target).square().sum(1)
            # dtime_tv_loss = torch.where(dtime.abs() < 1e-1, dtime_tv_loss, 0. * dtime_tv_loss)
            losses['dtime_tv_loss'] = dtime_tv_loss.mean()
            losses['tv_loss'] = (dtime_tv_loss / dtime.abs()).mean()
        else:
            mean_flow, dmean_flow__dstart_time = torch.autograd.functional.jvp(
                self,
                (x_start_time, start_time, end_time),
                v=(velocity, self.one.expand(start_time.shape), self.zero.expand(start_time.shape)),
                create_graph=True,
            )
            dmean_flow__dstart_time = dmean_flow__dstart_time.detach()
            target_mean_flow = velocity + dtime * dmean_flow__dstart_time

        losses['mean_flow_loss'] = (mean_flow - target_mean_flow).square().sum(1).mean()

        losses['loss'] = losses['mean_flow_loss'] + self.cfg.model.tv_loss_coeff * losses.get('dtime_tv_loss', 0.)

        return losses


@hydra.main(**mftv.utils.HYDRA_INIT)
def main(cfg):
    with conf.conf.Session() as db:
        cfg = conf.conf.orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
        db.commit()
        log.info('Command: python %s', ' '.join(sys.argv[:-1]))
        log.info(pprint.pformat(cfg))
        log.info('Output directory: %s', cfg.run_dir)

    rng = np.random.default_rng(mftv.utils.RNG_RANDBITS['DATASET'][cfg.rng_seed])
    (
        rng_dataset_start,
        rng_dataset_end,
        rng_mean_flow,
    ) = rng.spawn(3)

    dataset_start = mftv.datasets.get_dataset(cfg.dataset_start, rng)
    dataset_end = mftv.datasets.get_dataset(cfg.dataset_end, rng)

    pl.seed_everything(cfg.rng_seed)
    with pl.utilities.seed.isolate_rng():
        # model = mftv.model.MeanFlowModel(input_dim=2, output_dim=2, dim=256, n_hidden=2)
        model = mftv.model.MeanFlowModel(input_dim=cfg.dataset_start.dim, output_dim=cfg.dataset_end.dim, dim=64, n_hidden=4)

    mean_flow = MeanFlow(cfg, rng_mean_flow, dataset_start, dataset_end, model)

    logger = mftv.loggers.CSVLogger(cfg.run_dir, name=None)

    callbacks = [
        # mftv.callbacks.EMAWeightAveraging(),
        mftv.callbacks.ModelCheckpoint(
            dirpath=cfg.run_dir,
            filename='{epoch}',
            save_last='link',
            # monitor='loss',  # do not use 'loss' with mean flow loss, use some other metric
            # save_top_k=2,
            save_on_train_epoch_end=True,
            enable_version_counter=False,
        ),
        mftv.callbacks.LogStats(),
    ]
    trainer = pl.Trainer(
        accelerator=cfg.device,
        devices=1,
        logger=logger,
        max_steps=5_000,
        # max_epochs=50,
        # reload_dataloaders_every_n_epochs=1,
        deterministic=True,
        callbacks=callbacks,
        log_every_n_steps=50,
    )

    if cfg.fit:
        trainer.fit(mean_flow)


if __name__ == '__main__':
    last_override, run_dir = conf.conf.get_run_dir()
    conf.conf.set_run_dir(last_override, run_dir)
    main()
