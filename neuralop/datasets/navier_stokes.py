from pathlib import Path
from typing import List, Optional, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ..utils import UnitGaussianNormalizer
from .hdf5_dataset import H5pyDataset
from .zarr_dataset import ZarrDataset
from .tensor_dataset import TensorDataset
from .transforms import PositionalEmbedding, FutureEmbedding


def load_navier_stokes_zarr(
        data_path,
        n_train,
        batch_size,
        train_resolution=128,
        test_resolutions=(128, 256, 512, 1024),
        n_tests=(2000, 500, 500, 500),
        test_batch_sizes=(8, 4, 1),
        positional_encoding=True,
        grid_boundaries=((0, 1), (0, 1)),
        encode_input=True,
        encode_output=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
):
    data_path = Path(data_path)

    training_db = ZarrDataset(
        data_path / 'navier_stokes_1024_train.zarr',
        n_samples=n_train,
        resolution=train_resolution)
    training_db.generate_transforms(
        encode_input, positional_encoding, encode_output, grid_boundaries)

    train_loader = torch.utils.data.DataLoader(
        training_db,
        batch_size=batch_size,
        drop_last=True,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    test_loaders = dict()
    for (res, n_test, test_batch_size)\
            in zip(test_resolutions, n_tests, test_batch_sizes):
        print(
            f'Loading test db at resolution {res} with {n_test} samples and ' +
            f'batch-size={test_batch_size}')
        test_db = ZarrDataset(
            data_path /
            'navier_stokes_1024_test.zarr',
            n_samples=n_test,
            resolution=res,
            transform_x=training_db.transform_x,
            transform_y=training_db.transform_y)

        test_loaders[res] = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers)

    return train_loader, test_loaders, training_db.transform_y


def load_navier_stokes_hdf5(
        data_path,
        n_train,
        batch_size,
        train_resolution=128,
        test_resolutions=(128, 256, 512, 1024),
        n_tests=(2000, 500, 500, 500),
        test_batch_sizes=(8, 4, 1),
        positional_encoding=True,
        grid_boundaries=((0, 1), (0, 1)),
        encode_input=True,
        encode_output=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False
):
    data_path = Path(data_path)

    training_db = H5pyDataset(
        data_path / 'navier_stokes_1024_train.hdf5',
        n_samples=n_train,
        resolution=train_resolution)
    training_db.generate_transforms(
        encode_input, positional_encoding, encode_output, grid_boundaries)

    train_loader = torch.utils.data.DataLoader(
        training_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    test_loaders = dict()
    for (res, n_test, test_batch_size)\
        in zip(test_resolutions, n_tests, test_batch_sizes):
        print(
            f'Loading test db at resolution {res} with {n_test} samples and ' +
            f'batch-size={test_batch_size}')

        test_db = H5pyDataset(
            data_path / 'navier_stokes_1024_test.hdf5',
            n_samples=n_test,
            resolution=res,
            transform_x=training_db.transform_x,
            transform_y=training_db.transform_y)

        test_loaders[res] = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers)

    return train_loader, test_loaders, training_db.transform_y


EncodingEnum = Literal['channel-wise', 'pixel-wise']

def load_navier_stokes_pt(
    data_path,
    train_resolution,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions,
    grid_boundaries=((0, 1), (0, 1)),
    positional_encoding=True,
    encode_input=True,
    encode_output=True,
    encoding: EncodingEnum ='channel-wise',
    channel_dim=1,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
):
    """Load the Navier-Stokes dataset."""
    # assert train_resolution == 128,
    # 'Loading from pt only supported for train_resolution of 128'

    data = torch.load(
        Path(data_path)
        .joinpath(f'nsforcing_{train_resolution}_train.pt')
        .as_posix())
    x_train = data['x'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    y_train = data['y'][0:n_train, :, :].unsqueeze(channel_dim).clone()
    del data

    idx = test_resolutions.index(train_resolution)
    test_resolutions.pop(idx)
    n_test = n_tests.pop(idx)
    test_batch_size = test_batch_sizes.pop(idx)

    data = torch.load(
        Path(data_path)
        .joinpath(f'nsforcing_{train_resolution}_test.pt')
        .as_posix())
    x_test = data['x'][-n_test:, :, :].unsqueeze(channel_dim).clone()
    y_test = data['y'][-n_test:, :, :].unsqueeze(channel_dim).clone()
    del data

    if encode_input:
        input_encoder = _make_encoder(x_train, encoding)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        output_encoder = _make_encoder(y_train, encoding)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(
        x_train,
        y_train,
        transform_x=(
            PositionalEmbedding(grid_boundaries, 0)
            if positional_encoding
            else None))
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    test_db = TensorDataset(
        x_test,
        y_test,
        transform_x=(
            PositionalEmbedding(grid_boundaries, 0)
            if positional_encoding
            else None))
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    test_loaders = {train_resolution: test_loader}
    for (res, n_test, test_batch_size)\
            in zip(test_resolutions, n_tests, test_batch_sizes):
        print(
            f'Loading test db at resolution {res} with {n_test} samples and ' +
            f'batch-size={test_batch_size}')
        x_test, y_test = _load_navier_stokes_test_HR(
            data_path,
            n_test,
            resolution=res,
            channel_dim=channel_dim)
        if input_encoder is not None:
            x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(
            x_test,
            y_test,
            transform_x=(
                PositionalEmbedding(grid_boundaries, 0)
                if positional_encoding
                else None))
        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers)
        test_loaders[res] = test_loader

    return train_loader, test_loaders, output_encoder


def _load_navier_stokes_test_HR(data_path,
                                n_test,
                                resolution=256,
                                channel_dim=1):
    """Load the Navier-Stokes dataset."""
    if resolution == 128:
        downsample_factor = 8
    elif resolution == 256:
        downsample_factor = 4
    elif resolution == 512:
        downsample_factor = 2
    elif resolution == 1024:
        downsample_factor = 1
    else:
        raise ValueError(
            f'Invalid resolution, got {resolution}, expected one of ' +
            '[128, 256, 512, 1024].')

    data = torch.load(
        Path(data_path)
        .joinpath('nsforcing_1024_test1.pt')
        .as_posix())

    if not isinstance(n_test, int):
        n_samples = data['x'].shape[0]
        n_test = int(n_samples * n_test)

    downsample_slice = (
        slice(-n_test, None),
        slice(None, None, downsample_factor),
        slice(None, None, downsample_factor),
    )
    x_test = data['x'][downsample_slice]\
        .unsqueeze(channel_dim)\
        .clone()
    y_test = data['y'][downsample_slice]\
        .unsqueeze(channel_dim)\
        .clone()
    del data

    return x_test, y_test


def load_navier_stokes_temporal_pt(
    data_path: str,
    n_train: int,
    n_test: int,
    history_length: int,
    future_duration: int,
    train_batch_size: int,
    test_batch_size: int,
    downsampling_rate: int = 1,
    grid_boundaries=((0, 1), (0, 1)),
    positional_encoding: bool = True,
    future_encoding: bool = False,
    encode_input: bool = True,
    encode_output: bool = True,
    encoding: EncodingEnum = 'channel-wise',
    num_workers=2,
    pin_memory: bool = True,
    persistent_workers: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[UnitGaussianNormalizer]]:
    """Load a temporal Navier-Stokes dataset.

    The full training/testing dataset is expected to be of shape
    ``[N, T, X, Y]``, where:

    * N is the number of examples
    * T is the length of time each example contains
    * X, Y are the width, height of physical space

    This data will be partitioned into ``DataLoaders`` with shapes like:

    * ``train_loader[i].shape = [train_batch_size, history_length,
      X // downsampling_rate, Y // downsampling_rate]``
    * ``test_loader[j].shape = [test_batch_size, future_duration,
      X // downsampling_rate, Y // downsampling_rate]``

    Returns training ``DataLoader``, testing ``Dataloader``, and optional
    output encoder (as ``UnitGaussianNormalizer``) or ``None``.

    Parameters
    ----------
    data_path : str | Path
        Fully qualified path to target ``.pt`` or ``.pth`` PyTorch data file.
    n_train : int
        Number of training data points to include in returned ``train_loader``
    n_test : int
        Number of testing data points to include in returned ``test_loader``
    history_length : int
        Number of time steps into the past to include in the input tensors "X"
        in the training ``DataLoader``
    future_duration : int
        Number of time steps into the future to include in output tensors "Y"
        in the testing ``DataLoader``
    train_batch_size: int
        Size of batches to be used in training.
    test_batch_size: int
        Size of batches to be used in testing.
    downsampling_rate : int, optional; defaults to 1
        Step size to be used in x, y dimensions for training and testing data.
        A rate of 1 will use 100% of the given data (i.e. no downsampling),
        a rate of 4 will use 25% of data, etc.
    positional_encoding : bool, optional; defaults to ``True``
        Whether to add positional encoding the x_train and x_test data.
        Uses ``transforms.PositionalEmbedding``.
    grid_boundaries : Tuple[Tuple[int, int], Tuple[int, int]], optional
        By default is the unit square (i.e. ``((0, 1), (0, 1))``). Describes
        the lower (inclusive) and upper (exclusive) bounds to be used if
        ``positional_encoding==True``.
    future_encoding : bool, optional; defaults to ``False``
        Whether to add "future" encoding the x_train and x_test data using
        sequential integers ``[1 .. future_duration]``.
        Uses ``transforms.FutureEmbedding``.
    encode_input : bool, optional; defaults to ``True``
        Whether to normalize the input x_train, x_test using a Gaussian.
    encode_output : bool, optional; defaults to ``True``
        Whether to normalize the output y_train using a Gaussian.
    """
    future_end = history_length + future_duration
    data = torch.load(data_path)
    x_train_slice = (slice(n_train),  # Equivalent to:       [:n_train,
                     slice(history_length),  #                :history_length,
                     slice(None, None, downsampling_rate),  # ::sampling_rate,
                     slice(None, None, downsampling_rate))  # ::sampling_rate]
    x_train: torch.Tensor = data['u'][x_train_slice].clone()

    y_train_slice = (slice(n_train),  # Equivalent to:       [:n_train,
                     slice(history_length, future_end),  #    history:future,
                     slice(None, None, downsampling_rate),  # ::sampling_rate,
                     slice(None, None, downsampling_rate))  # ::sampling_rate]
    y_train: torch.Tensor = data['u'][y_train_slice].clone()

    x_test_slice = (slice(-n_test, None),  # Equivalent to: [-n_test:,
                    slice(history_length),  #                :history_length,
                    slice(None, None, downsampling_rate),  # ::sampling_rate,
                    slice(None, None, downsampling_rate))  # ::sampling_rate]
    x_test: torch.Tensor = data['u'][x_test_slice].clone()

    y_test_slice = (slice(-n_test, None),  # Equivalent to: [-n_test:,
                    slice(history_length, future_end),  #    history:future,
                    slice(None, None, downsampling_rate),  # ::sampling_rate,
                    slice(None, None, downsampling_rate))  # ::sampling_rate]
    y_test: torch.Tensor = data['u'][y_test_slice].clone()

    if encode_input:
        input_encoder = _make_encoder(x_train, encoding)
        x_train = input_encoder.encode(x_train)
        x_test = input_encoder.encode(x_test.contiguous())

    if encode_output:
        output_encoder = _make_encoder(y_train, encoding)
        y_train = output_encoder.encode(y_train)
    else:
        output_encoder = None

    train_transform_x = transforms.Compose([
        PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
        FutureEmbedding(future_duration, 1)
        if future_encoding
        else None
    ])
    train_set = TensorDataset(
        x_train,
        y_train,
        transform_x=train_transform_x)
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    test_transform_x = transforms.Compose([
        PositionalEmbedding(grid_boundaries, 0)
        if positional_encoding
        else None,
        FutureEmbedding(future_duration, 1)
        if future_encoding
        else None
    ])
    test_set = TensorDataset(
        x_test,
        y_test,
        transform_x=test_transform_x)
    test_loader = DataLoader(
        test_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    return train_loader, test_loader, output_encoder


def _make_encoder(
    target_tensor: torch.Tensor,
    encoding: EncodingEnum = 'channel-wise',
) -> UnitGaussianNormalizer:
    if encoding == 'channel-wise':
        reduce_dims: List[int] = list(range(target_tensor.ndim))
    elif encoding == 'pixel-wise':
        reduce_dims: List[int] = [0]
    else:
        raise ValueError(
            'Expected `encoding` to be one of "channel-wise", "pixel-wise";'
            f'got encoding={encoding}')

    return UnitGaussianNormalizer(
        target_tensor,
        reduce_dim=reduce_dims,
        verbose=False,
    )
