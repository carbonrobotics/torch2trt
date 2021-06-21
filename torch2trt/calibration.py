import torch
import tensorrt as trt
import os


if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    

class TensorBatchDataset():
    
    def __init__(self, tensors):
        self.tensors = tensors
    
    def __len__(self):
        return len(self.tensors[0])
    
    def __getitem__(self, idx):
        return [t[idx] for t in self.tensors]
    
    
class DatasetCalibrator(trt.IInt8Calibrator):
    
    def __init__(self, inputs, dataset, batch_size=1, algorithm=DEFAULT_CALIBRATION_ALGORITHM,
                 cache_input_path: str = None, cache_output_path: str = None):
        super(DatasetCalibrator, self).__init__()
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.cache_input_path = cache_input_path
        self.cache_output_path = cache_output_path
        
        # create buffers that will hold data batches
        self.buffers = []
        for tensor in inputs:
            size = (batch_size,) + tuple(tensor.shape[1:])
            buf = torch.zeros(size=size, dtype=tensor.dtype, device=tensor.device).contiguous()
            self.buffers.append(buf)
            
        self.count = 0
        
    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):
            
            for i in range(self.batch_size):
                
                idx = self.count % len(self.dataset) # roll around if not multiple of dataset
                inputs = self.dataset[idx]
                
                # copy data for (input_idx, dataset_idx) into buffer
                for buffer, tensor in zip(self.buffers, inputs):
                    buffer[i].copy_(tensor)
                
                self.count += 1
                
            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []
        
    def get_algorithm(self):
        return self.algorithm
    
    def get_batch_size(self):
        return self.batch_size
    
    def read_calibration_cache(self, *args, **kwargs):
        if self.cache_input_path is not None and os.path.exists(self.cache_input_path):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache, *args, **kwargs):
        if self.cache_input_path is not None:
            with open(self.cache_file, "wb") as f:
                f.write(cache)