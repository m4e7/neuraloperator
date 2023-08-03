from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import Normalizer, PositionalEmbedding


class H5pyDataset(Dataset):
    """PDE h5py dataset"""

    def __init__(
            self,
            data_path,
            resolution=128,
            transform_x=None,
            transform_y=None,
            n_samples=None):
        resolution_to_step = {128: 8, 256: 4, 512: 2, 1024: 1}
        try:
            subsample_step = resolution_to_step[resolution]
        except KeyError:
            raise ValueError(
                f'Got resolution={resolution}, ' +
                f'expected one of {resolution_to_step.keys()}')

        self.subsample_step = subsample_step
        self.data_path = data_path
        self._data = None
        self._x_mean: Optional[float] = None
        self._x_std: Optional[float] = None
        self._y_mean: Optional[float] = None
        self._y_std: Optional[float] = None
        self.transform_x = transform_x
        self.transform_y = transform_y

        if n_samples is not None:
            self.n_samples = n_samples
        else:
            with h5py.File(str(self.data_path), 'r') as f:
                self.n_samples = f['x'].shape[0]

    @property
    def data(self):
        if self._data is None:
            self._data = h5py.File(str(self.data_path), 'r')
        return self._data

    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    @property
    def x_mean(self):
        if self._x_mean is None:
            self._x_mean = self._attribute('x', 'mean')
        return self._x_mean

    @property
    def x_std(self):
        if self._x_std is None:
            self._x_std = self._attribute('x', 'std')
        return self._x_std

    @property
    def y_mean(self):
        if self._y_mean is None:
            self._y_mean = self._attribute('y', 'mean')
        return self._y_mean

    @property
    def y_std(self):
        if self._y_std is None:
            self._y_std = self._attribute('y', 'std')
        return self._y_std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if isinstance(idx, int):
            assert idx < self.n_samples, \
                f'Trying to access sample {idx} of dataset ' \
                f'with {self.n_samples} samples'
        else:
            for i in idx:
                assert i < self.n_samples, \
                    f'Trying to access sample {i} of dataset ' \
                    f'with {self.n_samples} samples'

        x = self.data['x'][idx, ::self.subsample_step, ::self.subsample_step]
        y = self.data['y'][idx, ::self.subsample_step, ::self.subsample_step]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        if self.transform_x:
            x = self.transform_x(x)

        if self.transform_y:
            y = self.transform_y(y)

        return {'x': x, 'y': y}

    def generate_transforms(
        self,
        encode_input,
        positional_encoding,
        encode_output,
        grid_boundaries=((0, 1), (0, 1)),
    ):
        transform_x = []
        transform_y = None

        if encode_input:
            transform_x.append(Normalizer(self.x_mean, self.x_std))

        if positional_encoding:
            transform_x.append(PositionalEmbedding(grid_boundaries, 0))

        if encode_output:
            transform_y = Normalizer(self.y_mean, self.y_std)

        self.transform_x = transforms.Compose(transform_x)
        self.transform_y = transform_y
