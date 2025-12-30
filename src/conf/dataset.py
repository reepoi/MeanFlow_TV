import omegaconf
from hydra_orm import orm
import sqlalchemy as sa


class Dataset(orm.InheritableTable):
    pass


class Gaussian(Dataset):
    dim: int = orm.make_field(orm.ColumnRequired(sa.Integer), default=2)
    std: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)


class Crescent(Dataset):
    inner_center: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.5)
    inner_radius: float = orm.make_field(orm.ColumnRequired(sa.Double), default=.6)
    outer_radius: float = orm.make_field(orm.ColumnRequired(sa.Double), default=1.)

    @property
    def dim(self):
        return 2
