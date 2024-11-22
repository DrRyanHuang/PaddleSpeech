import paddle
from pathlib import Path
from inspect import signature
from numpy.random import RandomState
from flatten_dict import flatten, unflatten
from typing import Callable, Dict, List, Union
from contextlib import contextmanager
import numpy as np

from . import util

from .audio_signal import AudioSignal

class AudioLoader:
    """Loads audio endlessly from a list of audio sources
    containing paths to audio files. Audio sources can be
    folders full of audio files (which are found via file
    extension) or by providing a CSV file which contains paths
    to audio files.

    Parameters
    ----------
    sources : List[str], optional
        Sources containing folders, or CSVs with
        paths to audio files, by default None
    weights : List[float], optional
        Weights to sample audio files from each source, by default None
    relative_path : str, optional
        Path audio should be loaded relative to, by default ""
    transform : Callable, optional
        Transform to instantiate alongside audio sample,
        by default None
    ext : List[str]
        List of extensions to find audio within each source by. Can
        also be a file name (e.g. "vocals.wav"). by default
        ``['.wav', '.flac', '.mp3', '.mp4']``.
    shuffle: bool
        Whether to shuffle the files within the dataloader. Defaults to True.
    shuffle_state: int
        State to use to seed the shuffle of the files.
    """

    def __init__(
        self,
        sources: List[str] = None,
        weights: List[float] = None,
        transform: Callable = None,
        relative_path: str = "",
        ext: List[str] = util.AUDIO_EXTENSIONS,
        shuffle: bool = True,
        shuffle_state: int = 0,
    ):
        self.audio_lists = util.read_sources(
            sources, relative_path=relative_path, ext=ext
        )

        self.audio_indices = [
            (src_idx, item_idx)
            for src_idx, src in enumerate(self.audio_lists)
            for item_idx in range(len(src))
        ]
        if shuffle:
            state = util.random_state(shuffle_state)
            state.shuffle(self.audio_indices)

        self.sources = sources
        self.weights = weights
        self.transform = transform

    def __call__(
        self,
        state,
        sample_rate: int,
        duration: float,
        loudness_cutoff: float = -40,
        num_channels: int = 1,
        offset: float = None,
        source_idx: int = None,
        item_idx: int = None,
        global_idx: int = None,
    ):
        if source_idx is not None and item_idx is not None:
            try:
                audio_info = self.audio_lists[source_idx][item_idx]
            except:
                audio_info = {"path": "none"}
        elif global_idx is not None:
            source_idx, item_idx = self.audio_indices[
                global_idx % len(self.audio_indices)
            ]
            audio_info = self.audio_lists[source_idx][item_idx]
        else:
            audio_info, source_idx, item_idx = util.choose_from_list_of_lists(
                state, self.audio_lists, p=self.weights
            )

        path = audio_info["path"]
        signal = AudioSignal.zeros(duration, sample_rate, num_channels)

        if path != "none":
            if offset is None:
                signal = AudioSignal.salient_excerpt(
                    path,
                    duration=duration,
                    state=state,
                    loudness_cutoff=loudness_cutoff,
                )
            else:
                signal = AudioSignal(
                    path,
                    offset=offset,
                    duration=duration,
                )

        if num_channels == 1:
            signal = signal.to_mono()
        signal = signal.resample(sample_rate)

        if signal.duration < duration:
            signal = signal.zero_pad_to(int(duration * sample_rate))

        for k, v in audio_info.items():
            signal.metadata[k] = v

        item = {
            "signal": signal,
            "source_idx": source_idx,
            "item_idx": item_idx,
            "source": str(self.sources[source_idx]),
            "path": str(path),
        }
        if self.transform is not None:
            item["transform_args"] = self.transform.instantiate(state, signal=signal)
        return item


class BaseTransform:
    """This is the base class for all transforms that are implemented
    in this library. Transforms have two main operations: ``transform``
    and ``instantiate``.

    ``instantiate`` sets the parameters randomly
    from distribution tuples for each parameter. For example, for the
    ``BackgroundNoise`` transform, the signal-to-noise ratio (``snr``)
    is chosen randomly by instantiate. By default, it chosen uniformly
    between 10.0 and 30.0 (the tuple is set to ``("uniform", 10.0, 30.0)``).

    ``transform`` applies the transform using the instantiated parameters.
    A simple example is as follows:

    >>> seed = 0
    >>> signal = ...
    >>> transform = transforms.NoiseFloor(db = ("uniform", -50.0, -30.0))
    >>> kwargs = transform.instantiate()
    >>> output = transform(signal.clone(), **kwargs)

    By breaking apart the instantiation of parameters from the actual audio
    processing of the transform, we can make things more reproducible, while
    also applying the transform on batches of data efficiently on GPU,
    rather than on individual audio samples.

    ..  note::
        We call ``signal.clone()`` for the input to the ``transform`` function
        because signals are modified in-place! If you don't clone the signal,
        you will lose the original data.

    Parameters
    ----------
    keys : list, optional
        Keys that the transform looks for when
        calling ``self.transform``, by default []. In general this is
        set automatically, and you won't need to manipulate this argument.
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0

    Examples
    --------

    >>> seed = 0
    >>>
    >>> audio_path = "tests/audio/spk/f10_script4_produced.wav"
    >>> signal = AudioSignal(audio_path, offset=10, duration=2)
    >>> transform = tfm.Compose(
    >>>     [
    >>>         tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
    >>>         tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
    >>>     ],
    >>> )
    >>>
    >>> kwargs = transform.instantiate(seed, signal)
    >>> output = transform(signal, **kwargs)

    """

    def __init__(self, keys: list = [], name: str = None, prob: float = 1.0):
        # Get keys from the _transform signature.
        tfm_keys = list(signature(self._transform).parameters.keys())

        # Filter out signal and kwargs keys.
        ignore_keys = ["signal", "kwargs"]
        tfm_keys = [k for k in tfm_keys if k not in ignore_keys]

        # Combine keys specified by the child class, the keys found in
        # _transform signature, and the mask key.
        self.keys = keys + tfm_keys + ["mask"]

        self.prob = prob

        if name is None:
            name = self.__class__.__name__
        self.name = name

    def _prepare(self, batch: dict):
        sub_batch = batch[self.name]

        for k in self.keys:
            assert k in sub_batch.keys(), f"{k} not in batch"

        return sub_batch

    def _transform(self, signal):
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        return {}

    @staticmethod
    def apply_mask(batch: dict, mask: paddle.Tensor):
        """Applies a mask to the batch.

        Parameters
        ----------
        batch : dict
            Batch whose values will be masked in the ``transform`` pass.
        mask : paddle.Tensor
            Mask to apply to batch.

        Returns
        -------
        dict
            A dictionary that contains values only where ``mask = True``.
        """

        masked_batch = {}
        for k, v in flatten(batch).items():
            if paddle.is_tensor(v) and v.ndim == 0:
                # dirty fix to prevent crash
                masked_batch[k] = v.unsqueeze(0)[mask].squeeze(0)
            else:
                masked_batch[k] = v[mask]
                
        return unflatten(masked_batch)

    def transform(self, signal: AudioSignal, **kwargs):
        """Apply the transform to the audio signal,
        with given keyword arguments.

        Parameters
        ----------
        signal : AudioSignal
            Signal that will be modified by the transforms in-place.
        kwargs: dict
            Keyword arguments to the specific transforms ``self._transform``
            function.

        Returns
        -------
        AudioSignal
            Transformed AudioSignal.

        Examples
        --------

        >>> for seed in range(10):
        >>>     kwargs = transform.instantiate(seed, signal)
        >>>     output = transform(signal.clone(), **kwargs)

        """
        tfm_kwargs = self._prepare(kwargs)
        mask = tfm_kwargs["mask"]

        if paddle.any(mask):
            tfm_kwargs = self.apply_mask(tfm_kwargs, mask)
            tfm_kwargs = {k: v for k, v in tfm_kwargs.items() if k != "mask"}
            signal[mask] = self._transform(signal[mask], **tfm_kwargs)

        return signal

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    def instantiate(
        self,
        state: RandomState = None,
        signal: AudioSignal = None,
    ):
        """Instantiates parameters for the transform.

        Parameters
        ----------
        state : RandomState, optional
            _description_, by default None
        signal : AudioSignal, optional
            _description_, by default None

        Returns
        -------
        dict
            Dictionary containing instantiated arguments for every keyword
            argument to ``self._transform``.

        Examples
        --------

        >>> for seed in range(10):
        >>>     kwargs = transform.instantiate(seed, signal)
        >>>     output = transform(signal.clone(), **kwargs)

        """
        state = util.random_state(state)

        # Not all instantiates need the signal. Check if signal
        # is needed before passing it in, so that the end-user
        # doesn't need to have variables they're not using flowing
        # into their function.
        needs_signal = "signal" in set(signature(self._instantiate).parameters.keys())
        kwargs = {}
        if needs_signal:
            kwargs = {"signal": signal}

        # Instantiate the parameters for the transform.
        params = self._instantiate(state, **kwargs)
        for k in list(params.keys()):
            v = params[k]
            if isinstance(v, (AudioSignal, paddle.Tensor, dict)):
                params[k] = v
            else:
                params[k] = paddle.to_tensor(v)
        mask = state.rand() <= self.prob
        params[f"mask"] = paddle.to_tensor(mask)

        # Put the params into a nested dictionary that will be
        # used later when calling the transform. This is to avoid
        # collisions in the dictionary.
        params = {self.name: params}

        return params

    def batch_instantiate(
        self,
        states: list = None,
        signal: AudioSignal = None,
    ):
        """Instantiates arguments for every item in a batch,
        given a list of states. Each state in the list
        corresponds to one item in the batch.

        Parameters
        ----------
        states : list, optional
            List of states, by default None
        signal : AudioSignal, optional
            AudioSignal to pass to the ``self.instantiate`` section
            if it is needed for this transform, by default None

        Returns
        -------
        dict
            Collated dictionary of arguments.

        Examples
        --------

        >>> batch_size = 4
        >>> signal = AudioSignal(audio_path, offset=10, duration=2)
        >>> signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])
        >>>
        >>> states = [seed + idx for idx in list(range(batch_size))]
        >>> kwargs = transform.batch_instantiate(states, signal_batch)
        >>> batch_output = transform(signal_batch, **kwargs)
        """
        kwargs = []
        for state in states:
            kwargs.append(self.instantiate(state, signal))
        kwargs = util.collate(kwargs)
        return kwargs


class Compose(BaseTransform):
    """Compose applies transforms in sequence, one after the other. The
    transforms are passed in as positional arguments or as a list like so:

    >>> transform = tfm.Compose(
    >>>     [
    >>>         tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
    >>>         tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
    >>>     ],
    >>> )

    This will convolve the signal with a room impulse response, and then
    add background noise to the signal. Instantiate instantiates
    all the parameters for every transform in the transform list so the
    interface for using the Compose transform is the same as everything
    else:

    >>> kwargs = transform.instantiate()
    >>> output = transform(signal.clone(), **kwargs)

    Under the hood, the transform maps each transform to a unique name
    under the hood of the form ``{position}.{name}``, where ``position``
    is the index of the transform in the list. ``Compose`` can nest
    within other ``Compose`` transforms, like so:

    >>> preprocess = transforms.Compose(
    >>>     tfm.GlobalVolumeNorm(),
    >>>     tfm.CrossTalk(),
    >>>     name="preprocess",
    >>> )
    >>> augment = transforms.Compose(
    >>>     tfm.RoomImpulseResponse(),
    >>>     tfm.BackgroundNoise(),
    >>>     name="augment",
    >>> )
    >>> postprocess = transforms.Compose(
    >>>     tfm.VolumeChange(),
    >>>     tfm.RescaleAudio(),
    >>>     tfm.ShiftPhase(),
    >>>     name="postprocess",
    >>> )
    >>> transform = transforms.Compose(preprocess, augment, postprocess),

    This defines 3 composed transforms, and then composes them in sequence
    with one another.

    Parameters
    ----------
    *transforms : list
        List of transforms to apply
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, *transforms: list, name: str = None, prob: float = 1.0):
        if isinstance(transforms[0], list):
            transforms = transforms[0]

        for i, tfm in enumerate(transforms):
            tfm.name = f"{i}.{tfm.name}"

        keys = [tfm.name for tfm in transforms]
        super().__init__(keys=keys, name=name, prob=prob)

        self.transforms = transforms
        self.transforms_to_apply = keys

    @contextmanager
    def filter(self, *names: list):
        """This can be used to skip transforms entirely when applying
        the sequence of transforms to a signal. For example, take
        the following transforms with the names ``preprocess, augment, postprocess``.

        >>> preprocess = transforms.Compose(
        >>>     tfm.GlobalVolumeNorm(),
        >>>     tfm.CrossTalk(),
        >>>     name="preprocess",
        >>> )
        >>> augment = transforms.Compose(
        >>>     tfm.RoomImpulseResponse(),
        >>>     tfm.BackgroundNoise(),
        >>>     name="augment",
        >>> )
        >>> postprocess = transforms.Compose(
        >>>     tfm.VolumeChange(),
        >>>     tfm.RescaleAudio(),
        >>>     tfm.ShiftPhase(),
        >>>     name="postprocess",
        >>> )
        >>> transform = transforms.Compose(preprocess, augment, postprocess)

        If we wanted to apply all 3 to a signal, we do:

        >>> kwargs = transform.instantiate()
        >>> output = transform(signal.clone(), **kwargs)

        But if we only wanted to apply the ``preprocess`` and ``postprocess``
        transforms to the signal, we do:

        >>> with transform_fn.filter("preprocess", "postprocess"):
        >>>     output = transform(signal.clone(), **kwargs)

        Parameters
        ----------
        *names : list
            List of transforms, identified by name, to apply to signal.
        """
        old_transforms = self.transforms_to_apply
        self.transforms_to_apply = names
        yield
        self.transforms_to_apply = old_transforms

    def _transform(self, signal, **kwargs):
        for transform in self.transforms:
            if any([x in transform.name for x in self.transforms_to_apply]):
                signal = transform(signal, **kwargs)
        return signal

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        parameters = {}
        for transform in self.transforms:
            parameters.update(transform.instantiate(state, signal=signal))
        return parameters

    def __getitem__(self, idx):
        return self.transforms[idx]

    def __len__(self):
        return len(self.transforms)

    def __iter__(self):
        for transform in self.transforms:
            yield transform


class Choose(Compose):
    """Choose logic is the same as :py:func:`audiotools.data.transforms.Compose`,
    but instead of applying all the transforms in sequence, it applies just a single transform,
    which is chosen for each item in the batch.

    Parameters
    ----------
    *transforms : list
        List of transforms to apply
    weights : list
        Probability of choosing any specific transform.
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0

    Examples
    --------

    >>> transforms.Choose(tfm.LowPass(), tfm.HighPass())
    """

    def __init__(
        self,
        *transforms: list,
        weights: list = None,
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(*transforms, name=name, prob=prob)

        if weights is None:
            _len = len(self.transforms)
            weights = [1 / _len for _ in range(_len)]
        self.weights = np.array(weights)

    def _instantiate(self, state: RandomState, signal: AudioSignal = None):
        kwargs = super()._instantiate(state, signal)
        tfm_idx = list(range(len(self.transforms)))
        tfm_idx = state.choice(tfm_idx, p=self.weights)
        one_hot = []
        for i, t in enumerate(self.transforms):
            mask = kwargs[t.name]["mask"]
            if mask.item():
                kwargs[t.name]["mask"] = paddle.to_tensor(i == tfm_idx)
            one_hot.append(kwargs[t.name]["mask"])
        kwargs["one_hot"] = one_hot
        return kwargs


class Identity(BaseTransform):
    """This transform just returns the original signal."""

    pass


class VolumeNorm(BaseTransform):
    """Normalizes the volume of the excerpt to a specified decibel.

    Uses :py:func:`audiotools.core.effects.EffectMixin.normalize`.

    Parameters
    ----------
    db : tuple, optional
        dB to normalize signal to, by default ("const", -24)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        db: tuple = ("const", -24),
        name: str = None,
        prob: float = 1.0,
    ):
        super().__init__(name=name, prob=prob)

        self.db = db

    def _instantiate(self, state: RandomState):
        return {"db": util.sample_from_dist(self.db, state)}

    def _transform(self, signal, db):
        return signal.normalize(db)
    

class RescaleAudio(BaseTransform):
    """Rescales the audio so it is in between ``-val`` and ``val``
    only if the original audio exceeds those bounds. Useful if
    transforms have caused the audio to clip.

    Uses :py:func:`audiotools.core.effects.EffectMixin.ensure_max_of_audio`.

    Parameters
    ----------
    val : float, optional
        Max absolute value of signal, by default 1.0
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(self, val: float = 1.0, name: str = None, prob: float = 1):
        super().__init__(name=name, prob=prob)

        self.val = val

    def _transform(self, signal):
        return signal.ensure_max_of_audio(self.val)
    

class SpectralTransform(BaseTransform):
    """Spectral transforms require STFT data to exist, since manipulations
    of the STFT require the spectrogram. This just calls ``stft`` before
    the transform is called, and calls ``istft`` after the transform is
    called so that the audio data is written to after the spectral
    manipulation.
    """

    def transform(self, signal, **kwargs):
        signal.stft()
        super().transform(signal, **kwargs)
        signal.istft()
        return signal
    

class ShiftPhase(SpectralTransform):
    """Shifts the phase of the audio.

    Uses :py:func:`audiotools.core.dsp.DSPMixin.shift)phase`.

    Parameters
    ----------
    shift : tuple, optional
        How much to shift phase by, by default ("uniform", -np.pi, np.pi)
    name : str, optional
        Name of this transform, used to identify it in the dictionary
        produced by ``self.instantiate``, by default None
    prob : float, optional
        Probability of applying this transform, by default 1.0
    """

    def __init__(
        self,
        shift: tuple = ("uniform", -np.pi, np.pi),
        name: str = None,
        prob: float = 1,
    ):
        super().__init__(name=name, prob=prob)
        self.shift = shift

    def _instantiate(self, state: RandomState):
        return {"shift": util.sample_from_dist(self.shift, state)}

    def _transform(self, signal, shift):
        return signal.shift_phase(shift)
