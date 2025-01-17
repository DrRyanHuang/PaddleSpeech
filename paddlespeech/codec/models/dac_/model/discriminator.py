import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddlespeech.audiotools import AudioSignal
from paddlespeech.audiotools import ml
from paddlespeech.audiotools import STFTParams


def WNConv1d(*args, **kwargs):
    act = kwargs.pop("act", True)
    model = nn.Conv1D(*args, **kwargs)
    conv = nn.utils.weight_norm(model)
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    model = nn.Conv2D(*args, **kwargs)
    conv = nn.utils.weight_norm(model)
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MPD(nn.Layer):
    def __init__(self, period):
        super(MPD, self).__init__()
        self.period = period
        self.convs = nn.LayerList([
            WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
            WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
        ])
        self.conv_post = WNConv2d(
            1024, 1, kernel_size=(3, 1), padding=(1, 0), act=False)

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(
            x, (0, self.period - t % self.period),
            mode="reflect",
            data_format="NCL")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        # x = rearrange(x, "b c (l p) -> b c l p", p=self.period)
        b, c, lp = x.shape
        l = lp // self.period
        p = self.period
        x = x.reshape([b, c, l, p])

        for layer in self.convs:
            x = layer(x)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class MSD(nn.Layer):
    def __init__(self, rate: int=1, sample_rate: int=44100):
        super(MSD, self).__init__()
        self.convs = nn.LayerList([
            WNConv1d(1, 16, 15, 1, padding=7),
            WNConv1d(16, 64, 41, 4, groups=4, padding=20),
            WNConv1d(64, 256, 41, 4, groups=16, padding=20),
            WNConv1d(256, 1024, 41, 4, groups=64, padding=20),
            WNConv1d(1024, 1024, 41, 4, groups=256, padding=20),
            WNConv1d(1024, 1024, 5, 1, padding=2),
        ])
        self.conv_post = WNConv1d(1024, 1, 3, 1, padding=1, act=False)
        self.sample_rate = sample_rate
        self.rate = rate

    def forward(self, x):
        x = AudioSignal(x, self.sample_rate)
        x.resample(self.sample_rate // self.rate)
        x = x.audio_data

        fmap = []

        for l in self.convs:
            x = l(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]


class MRD(nn.Layer):
    def __init__(
            self,
            window_length: int,
            hop_factor: float=0.25,
            sample_rate: int=44100,
            bands: list=BANDS, ):
        """Complex multi-band spectrogram discriminator.
        Parameters
        ----------
        window_length : int
            Window length of STFT.
        hop_factor : float, optional
            Hop factor of the STFT, defaults to ``0.25 * window_length``.
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run discriminator over.
        """
        super(MRD, self).__init__()

        self.window_length = window_length
        self.hop_factor = hop_factor
        self.sample_rate = sample_rate
        self.stft_params = STFTParams(
            window_length=window_length,
            hop_length=int(window_length * hop_factor),
            match_stride=True, )

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch = 32

        def convs():
            return nn.LayerList([
                WNConv2d(2, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ])

        self.band_convs = nn.LayerList(
            [convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(
            ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        x = AudioSignal(x, self.sample_rate, stft_params=self.stft_params)
        x = paddle.as_real(x.stft())
        # x = rearrange(x, "b 1 f t c -> (b 1) c t f")
        x = x.transpose([0, 1, 4, 3, 2]).flatten(stop_axis=1)

        # Split into bands
        x_bands = [x[..., b[0]:b[1]] for b in self.bands]
        return x_bands

    def forward(self, x):
        x_bands = self.spectrogram(x)
        fmap = []

        x = []
        for band, stack in zip(x_bands, self.band_convs):
            for layer in stack:
                band = layer(band)
                fmap.append(band)
            x.append(band)

        x = paddle.concat(x, axis=-1)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class Discriminator(ml.BaseModel):
    def __init__(
            self,
            rates: list=[],
            periods: list=[2, 3, 5, 7, 11],
            fft_sizes: list=[2048, 1024, 512],
            sample_rate: int=44100,
            bands: list=BANDS, ):
        """Discriminator that combines multiple discriminators.

        Parameters
        ----------
        rates : list, optional
            sampling rates (in Hz) to run MSD at, by default []
            If empty, MSD is not used.
        periods : list, optional
            periods (of samples) to run MPD at, by default [2, 3, 5, 7, 11]
        fft_sizes : list, optional
            Window sizes of the FFT to run MRD at, by default [2048, 1024, 512]
        sample_rate : int, optional
            Sampling rate of audio in Hz, by default 44100
        bands : list, optional
            Bands to run MRD at, by default `BANDS`
        """
        super(Discriminator, self).__init__()
        discs = []
        discs += [MPD(p) for p in periods]
        discs += [MSD(r, sample_rate=sample_rate) for r in rates]
        discs += [
            MRD(f, sample_rate=sample_rate, bands=bands) for f in fft_sizes
        ]
        self.discriminators = nn.LayerList(discs)

    def preprocess(self, y):
        # Remove DC offset
        y = y - y.mean(axis=-1, keepdim=True)
        # Peak normalize the volume of input audio
        y = 0.8 * y / (paddle.abs(y).max(axis=-1, keepdim=True)[0] + 1e-9)
        return y

    def forward(self, x):
        x = self.preprocess(x)
        fmaps = [d(x) for d in self.discriminators]
        return fmaps


if __name__ == "__main__":
    disc = Discriminator()
    x = paddle.zeros([1, 1, 44100])
    results = disc(x)
    for i, result in enumerate(results):
        print(f"disc{i}")
        for i, r in enumerate(result):
            print(r.shape, r.mean(), r.min(), r.max())
        print()
