from __future__ import absolute_import

from .Video_ResNet import ME_ResNet50,MultiLoss_ResNet50,CoordAtt_ResNet50,Baseline
from .integrate import ME_CoordAtt_MultiLoss_ResNet50

__factory = {
    'me': ME_ResNet50,
    'multiloss':MultiLoss_ResNet50,
    'coordatt':CoordAtt_ResNet50,
    'coordatt_me_multiloss':ME_CoordAtt_MultiLoss_ResNet50,
    'baseline':Baseline,

}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
