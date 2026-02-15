# Vector Fitting (VF) – Python Implementation with Parallel Processing

A fast and robust **Vector Fitting** implementation in Python for rational approximation of frequency-domain data (e.g. S-parameters, admittance, impedance, transfer functions).

This version includes:

- Parallel matrix construction using `ProcessPoolExecutor`
- Proper handling of real poles + complex conjugate pairs
- Automatic generation of initial poles (log-spaced or user-defined resonant frequencies)
- Stability enforcement (poles forced to left half-plane)
- Support for both real and imaginary parts in least-squares fitting

## Features

- Parallelized bottleneck (`build_A`) → significantly faster on multi-core systems
- Clean separation of real/complex-conjugate pole treatment
- User-friendly interactive mode for quick testing
- Synthetic example included (3 resonances + linear term + noise)
- Final fit quality evaluation (relative error)

## Requirements

```text
Python 3.8+
numpy
(optional) tqdm, scipy    # only if you extend the code
