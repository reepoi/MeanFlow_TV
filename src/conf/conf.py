from dataclasses import field
from pathlib import Path
import sys
from typing import Any, List

import hydra
import hydra_orm.utils
import omegaconf
from omegaconf import OmegaConf
import sqlalchemy as sa
from hydra_orm import orm

import conf.dataset
import conf.model
import mftv.utils


def get_engine(dir=str(mftv.utils.DIR_ROOT), name='runs'):
    return sa.create_engine(f'sqlite+pysqlite:///{dir}/{name}.sqlite')


engine = get_engine()
orm.create_all(engine)
Session = sa.orm.sessionmaker(engine)


def get_run_dir(hydra_init=mftv.utils.HYDRA_INIT, commit=True, engine_name='runs'):
    if '-m' in sys.argv or '--multirun' in sys.argv:
        raise ValueError("The flags '-m' and '--multirun' are not supported. Use GNU parallel instead.")
    with hydra.initialize(version_base=hydra_init['version_base'], config_path=hydra_init['config_path']):
        last_override = None
        overrides = []
        for i, a in enumerate(sys.argv):
            if '=' in a:
                overrides.append(a)
                last_override = i
        cfg = hydra.compose(hydra_init['config_name'], overrides=overrides)
        engine = get_engine(name=engine_name)
        orm.create_all(engine)
        with sa.orm.Session(engine, expire_on_commit=False) as db:
            cfg = orm.instantiate_and_insert_config(db, OmegaConf.to_container(cfg, resolve=True))
            # if commit and '-c' not in sys.argv:
            if commit:
                db.commit()
                cfg.run_dir.mkdir(exist_ok=True)
            return last_override, str(cfg.run_dir)


def set_run_dir(last_override, run_dir):
    run_dir_override = f'hydra.run.dir={run_dir}'
    if last_override is None:
        sys.argv.append(run_dir_override)
    else:
        sys.argv.insert(last_override + 1, run_dir_override)


class Conf(orm.InheritableTable):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    root_dir: str = field(default=str(mftv.utils.DIR_ROOT.resolve()))
    out_dir: str = field(default=str((mftv.utils.DIR_ROOT/'..'/'..'/'out'/'MeanFlowTV').resolve()))
    run_subdir: str = field(default='runs')
    prediction_filename: str = field(default='output')
    device: str = field(default='cuda')

    alt_id: str = orm.make_field(orm.ColumnRequired(sa.String(8), index=True, unique=True), init=False, omegaconf_ignore=True)
    rng_seed: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2376999025)

    @property
    def run_dir(self):
        return Path(self.out_dir)/self.run_subdir/self.alt_id


sa.event.listens_for(Conf, 'before_insert', propagate=True)(
    hydra_orm.utils.set_attr_to_func_value(Conf, Conf.alt_id.key, hydra_orm.utils.generate_random_string)
)


class TrainingAndEvaluation(Conf):
    defaults: List[Any] = hydra_orm.utils.make_defaults_list([
        dict(dataset_start=omegaconf.MISSING),
        dict(dataset_end=omegaconf.MISSING),
        dict(model=omegaconf.MISSING),
        '_self_',
    ])
    fit: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=True)
    predict: bool = orm.make_field(orm.ColumnRequired(sa.Boolean), default=False)

    dataset_start = orm.OneToManyField(conf.dataset.Dataset, required=True, default=omegaconf.MISSING, column_name='DatasetStart')
    dataset_end = orm.OneToManyField(conf.dataset.Dataset, required=True, default=omegaconf.MISSING, column_name='DatasetEnd')
    model = orm.OneToManyField(conf.model.Model, required=True, default=omegaconf.MISSING)

    # def get_model(self):
    #     if isinstance(self.model, Trained):
    #         return self.model.conf.model
    #     else:
    #         return self.model


# class Trained(conf.model.Model):
#     conf = orm.OneToManyField(Conf, default=omegaconf.MISSING, enforce_element_type=False)
#     ckpt_filename: str = orm.make_field(orm.ColumnRequired(sa.String(len('epoch_####.ckpt'))), default='last.ckpt')
#
#     @staticmethod
#     def transform_conf(session, conf_alt_id):
#         if conf_alt_id == omegaconf.MISSING:
#             raise ValueError('Please set a conf alt_id with model.conf=<conf_alt_id>.')
#         conf = session.query(Conf).filter_by(alt_id=conf_alt_id).first()
#         assert conf is not None
#         return conf

orm.store_config(TrainingAndEvaluation)
orm.store_config(conf.model.MeanFlow, group=TrainingAndEvaluation.model.key)
for group in (TrainingAndEvaluation.dataset_start.key, TrainingAndEvaluation.dataset_end.key):
    orm.store_config(conf.dataset.Gaussian, group=group)
    orm.store_config(conf.dataset.Crescent, group=group)
