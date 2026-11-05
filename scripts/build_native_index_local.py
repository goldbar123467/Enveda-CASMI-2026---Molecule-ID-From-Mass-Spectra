"""Build the native-spectrum row index with one low-priority CPU while CUDA trains.

This streams model-input hashes, not a spectral-library search or simulator.
The resulting index must agree with the independently built Kaggle index.
"""
import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    process = kernel.GetCurrentProcess()
    assert kernel.SetPriorityClass(process, 0x00004000)  # Below normal, this process only.
    assert kernel.SetProcessAffinityMask(process, 1 << min((os.cpu_count() or 1) - 1, 7))

from pathlib import Path
import pyarrow as pa
from cloud_native_pair import native_index

pa.set_cpu_count(1)
pa.set_io_thread_count(1)
root = Path(__file__).resolve().parents[1]
native_index(root / 'outputs/local/n0/cache',
             root / 'outputs/forks/p04-native-unseen/native_training_spectra.parquet',
             root / 'outputs/local/kit', root / 'outputs/local/paired-native/native_rows.npy')
print('[0] DONE native row index prepared locally under a one-core limit', flush=True)
