import omegaconf
from hydra_orm import orm
import sqlalchemy as sa


class Model(orm.InheritableTable):
    batch_size: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=omegaconf.MISSING)
    lr: float = orm.make_field(orm.ColumnRequired(sa.Double), default=omegaconf.MISSING)


class MeanFlow(Model):
    tv_loss_coeff: float = orm.make_field(orm.ColumnRequired(sa.Double), default=0.)
