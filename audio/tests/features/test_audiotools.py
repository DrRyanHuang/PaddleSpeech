import paddle
from pathlib import Path

import audiotools
from audiotools.audio_signal import AudioSignal
from audiotools import util
from audiotools import transforms as trf


non_deterministic_transforms = ["TimeNoise", "FrequencyNoise"]


def _compare_transform(transform_name, signal):
    regression_data = Path(f"tests/regression/transforms/{transform_name}.wav")
    regression_data.parent.mkdir(exist_ok=True, parents=True)

    if regression_data.exists():
        regression_signal = AudioSignal(regression_data)
        assert paddle.allclose(
            signal.audio_data, regression_signal.audio_data, atol=1e-4
        )
    else:
        signal.write(regression_data)

def _test_transform(transform_name: str):
    seed = 0
    util.seed(seed)
    transform_cls = getattr(trf, transform_name)
    

    kwargs = {}
    if transform_name == "BackgroundNoise":
        kwargs["sources"] = ["tests/features/testdata/audiotools/noises.csv"]

    audio_path = "tests/features/testdata/audiotools/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = transform_cls(prob=1.0, **kwargs)

    kwargs = transform.instantiate(seed, signal)
    for k in kwargs[transform_name]:
        assert k in transform.keys

    output = transform(signal, **kwargs)
    assert isinstance(output, AudioSignal)

    _compare_transform(transform_name, output)

    if transform_name in non_deterministic_transforms:
        return

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])
    signal_batch.metadata["loudness"] = AudioSignal(audio_path).ffmpeg_loudness().item()

    states = [seed + idx for idx in list(range(batch_size))]
    kwargs = transform.batch_instantiate(states, signal_batch)
    batch_output = transform(signal_batch, **kwargs)

    assert batch_output[0] == output

    ## Test that you can apply transform with the same args twice.
    signal = AudioSignal(audio_path, offset=10, duration=2)
    kwargs = transform.instantiate(seed, signal)
    output_a = transform(signal.clone(), **kwargs)
    output_b = transform(signal.clone(), **kwargs)

    assert output_a == output_b


def test_compose_basic():
    seed = 0

    audio_path = "tests/features/testdata/audiotools/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Compose(
        [
            tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
        ],
    )

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal, **kwargs)

    _compare_transform("Compose", output)

    assert isinstance(transform[0], tfm.RoomImpulseResponse)
    assert isinstance(transform[1], tfm.BackgroundNoise)
    assert len(transform) == 2

    # Make sure __iter__ works
    for _tfm in transform:
        pass


def test_transform():
    transform_list = []
    for t in transform_list:
        _test_transform(t)


def test_choose_basic():
    seed = 0
    audio_path = "tests/features/testdata/audiotools/spk/f10_script4_produced.wav"
    signal = AudioSignal(audio_path, offset=10, duration=2)
    transform = tfm.Choose(
        [
            tfm.RoomImpulseResponse(sources=["tests/audio/irs.csv"]),
            tfm.BackgroundNoise(sources=["tests/audio/noises.csv"]),
        ]
    )

    kwargs = transform.instantiate(seed, signal)
    output = transform(signal.clone(), **kwargs)

    _compare_transform("Choose", output)

    transform = tfm.Choose(
        [
            MulTransform(0.0),
            MulTransform(2.0),
        ]
    )
    targets = [signal.clone() * 0.0, signal.clone() * 2.0]

    for seed in range(10):
        kwargs = transform.instantiate(seed, signal)
        output = transform(signal.clone(), **kwargs)

        assert any([output == target for target in targets])

    # Test that if you make a batch of signals and call it,
    # the first item in the batch is still the same as above.
    batch_size = 4
    signal = AudioSignal(audio_path, offset=10, duration=2)
    signal_batch = AudioSignal.batch([signal.clone() for _ in range(batch_size)])

    states = [seed + idx for idx in list(range(batch_size))]
    kwargs = transform.batch_instantiate(states, signal_batch)
    batch_output = transform(signal_batch, **kwargs)

    for nb in range(batch_size):
        assert batch_output[nb] in targets


if __name__ == "__main__":
    test_transform()
    test_choose_basic()
    test_compose_basic()
