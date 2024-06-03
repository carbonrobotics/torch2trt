import torch
import tensorrt as trt
import os
from .flattener import Flattener
from .version_utils import trt_version

__all__ = [
    'DEFAULT_CALIBRATION_ALGORITHM',
    'DatasetCalibrator'
]


if trt_version() >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class DatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self, dataset, algorithm=DEFAULT_CALIBRATION_ALGORITHM, batch_size=1, cache_file=None, flattener=None):
        super(DatasetCalibrator, self).__init__()
        self.dataset = dataset
        self.algorithm = algorithm
        self.count = 0
        self.cache_file = cache_file
        if flattener is None:
            flattener = Flattener.from_value(dataset[0])
        self.flattener = flattener
        self.batch_size = batch_size
        self.tensors = []

    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            self.tensors = []
            for _ in range(self.get_batch_size()):
                datapoint = self.flattener.flatten(self.dataset[self.count])
                for j in range(len(datapoint)):
                    if len(self.tensors) <= j:
                        self.tensors.append([datapoint[j]])
                    else:
                        self.tensors[j].append(datapoint[j])
            self.tensors = [torch.cat(tensor_list, dim=0) for tensor_list in self.tensors]
            bindings = [int(t.data_ptr()) for t in self.tensors]
            self.count += 1
            return bindings
        else:
            return []

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self, *args, **kwargs):
        if (self.cache_file is not None) and os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
                
    def write_calibration_cache(self, cache, *args, **kwargs):
        if self.cache_file is not None:
            with open(self.cache_file, 'wb') as f:
                f.write(cache)
