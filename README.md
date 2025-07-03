# Multi-Stage FNO training

This repo contains code for training the multi-stage FNO model described in the paper [Reducing Frequency Bias of Fourier Neural Operators in 3D Seismic Wavefield Simulations Through Multi-Stage Training](https://arxiv.org/abs/2503.02023). The coce has been testd on LLNL's (Tuolumne)[https://hpc.llnl.gov/hardware/compute-platforms/tuolumne] computing platform with 2 nodes and 8 GPUs. 

***
## [Content](#Content)

* 01_FNO-SingleFrequency-multistage.py - script to train the multi-stage FNO.
* 02_prepare_residule-stage2.py - script to prepare the residual for training the stage 2 model
* 03_run_test_SingleFrequency_multistage.py - script to run the test by combining the model from stage 1 and stage 2
* run.sh - bash script to set up environment variables and submit job to the computing platform
* utils/ - folder contains the functions used by the main scripts
* models/ - folder contains the FNO model for 3d data
* demo_data/*.npz - numpy zipped arrays, contains 5 simulation data at single frequency (2Hz) with naming sim_x_y.npz, where x is the index of the simulation, y is the incremental number for all the different frequencies. Since each simulation is approximated by using 11 frequencies, thus 0, 11, 22, 33, 44 indicate these are all for the lowest frequency (2Hz).
* requirements.txt - Python packages used to run the code.

***
## [License](#license)

This code is provided under the [MIT License](LICENSE.txt).

```text
 Copyright (c) 2025 Qingkai Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

***
## [Disclaimer](#disclaimer)
```text
  This work was performed under the auspices of the U.S. Department of Energy
  by Lawrence Livermore National Laboratory under Contract DE-AC52-07NA27344.
```

``LLNL-CODE-2007941``
