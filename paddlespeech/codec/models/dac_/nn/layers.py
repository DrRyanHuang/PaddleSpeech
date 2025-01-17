import paddle
import paddle.nn as nn
from paddle.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1D(*args, **kwargs), name='weight', dim=1)


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(
        nn.Conv1DTranspose(*args, **kwargs), name='weight', dim=1)


def snake(x, alpha):
    shape = x.shape
    x = x.reshape([shape[0], shape[1], -1])
    x = x + (alpha + 1e-9).reciprocal() * paddle.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Layer):
    def __init__(self, channels):
        super(Snake1d, self).__init__()
        self.alpha = self.create_parameter(
            shape=[1, channels, 1],
            default_initializer=nn.initializer.Constant(1.0))

    def forward(self, x):
        return snake(x, self.alpha)
