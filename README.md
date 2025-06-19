# HypoNet Nankai

## Overview

This repository contains Python implementations for HypoNet Nankai ([Agata+2025](https://doi.org/10.48550/arXiv.2411.04667)), a rapid hypocenter determination tool for the Nankai Trough subduction zone using physics-informed neural network (PINN). 

## Features

- Physics-Informed Neural Network (PINN) based hypocenter determination and travel time inference for P-wave
- 3D P-wave velocity structure for the Nankai Trough region ([Nakanishi+2018](https://doi.org/10.1130/2018.2534(04))) is incorporated. Currently, only events and stations in the offshore region are supported (see Figure 2 in Agata+2025).
- Both the command-line interface for hypocenter determination and python module for travel time inference are available
- We assume usage of HypoNet Nankai in CPU. However, the module also works in GPU.

## Installation

### Prerequisites

- Python >= 3.7
- pip (Python package installer)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourname/hyponet_project.git
cd hyponet_project
```

2. Install the package:
```bash
pip install -e .
```

### Dependencies

The project requires the following main dependencies:
- numpy
- torch >= 2.0.0 (CPU version is sufficient)
- matplotlib
- scipy
- netCDF4
- pymap3d

All dependencies will be automatically installed during the package installation.

## Usage

### Command Line Interface

The package provides a command-line interface for hypocenter determination. The main command is `hyponetn_run` with the following options:

```bash
hyponetn_run [options]
```

#### Command Line Options

- `--src_s`: Start event index (default: 0)
- `--src_e`: End event index (default: 0)
- `--eventdir`: Directory containing event files (required)
- `--outputdir`: Directory to save output files (required)
- `--dry_run`: Run the script in dry run mode (the program will only check if the receiver coordinates are valid. Hypocenter will not be determined)






#### Input & Output File Format

The input files should be located in the specified `eventdir` with the naming format:
```
eventdir/event.XXXXXXXX.txt
```


The output files are saved in the specified `outputdir` with the naming format:
```
outputdir/result.XXXXXXXX.txt
```
In both input and output file names, XXXXXXXX is the event index starting from 0 padded with zeros.

We supports two types of input file formats:

1. **Table style**

   Usually, using this style is recommended. 

   *Input file format:*

   Example of input file format with five stations:
   ```
   # index: 0 source_id: 0 (for your memory)
   nan 11.71 0.12 nan nan 32.97 136.65 0.14 0.
   nan 9.55 0.12 nan nan 32.97 136.86 0. 0.
   nan 11.32 0.12 nan nan 32.61 136.86 0.33 0.
   nan 10.54 0.11 nan nan 32.76 136.00 0. 0.
   nan 6.77 0.12 nan nan 33.06 136.17 0. 0.
   ```

  
   - Column 1: Station ID (not used by the code)
   - Column 2: P-wave travel time (seconds)
   - Column 3: P-wave travel time error (seconds)
   - Column 4: S-wave travel time (not used)
   - Column 5: S-wave travel time error (not used)
   - Column 6: Station latitude (degrees)
   - Column 7: Station longitude (degrees)
   - Column 8: P-wave station correction (seconds, added to observed travel time)
   - Column 9: S-wave station correction (not used)

   *Output file format:*
   ```
   #lon lat depth lon_err lat_err depth_err
   #Covariance matrix: xx xy xz yy yz zz
   [longitude] [latitude] [depth] [longitude_error] [latitude_error] [depth_error]
   [cov_xx] [cov_xy] [cov_xz] [cov_yy] [cov_yz] [cov_zz]
   ```

2. **"hypomh"-like style**

   "hypomh" is a code developed by Hirata & Matsu'ura (1987). The user of hypomh can readily use the same input format.

   Hirata, N., & Matsu'ura, M. (1987). Maximum-likelihood estimation of hypocenter with origin time eliminated using nonlinear inversion technique. Physics of the Earth and Planetary Interiors, 47, 50-61.

   Example of input file format with five stations:

   *Input file format:*
   ```
   #24/07/31 04:17                   24/07/23 16:42:00 (for your memory)
   01 . 11.71 0.12 - - 0. - 32.97 136.65 -1375 0.14
   02 .  9.55 0.12 - - 0. - 32.97 136.86 -48 
   03 . 11.32 0.12 - - 0. - 32.61 136.86 -1044 0.33
   04 . 10.54 0.11 - - 0. - 32.76 136.00 -1023
   05 .  6.77 0.12 - - 0. - 32.76 136.00 -1000 
   ```


   - Column 1: Station code (not used)
   - Column 2: P-wave first-motion polarity (not used)
   - Column 3: P-wave arrival time (seconds)
   - Column 4: P-wave arrival time error (seconds)
   - Column 5: S-wave arrival time (not used)
   - Column 6: S-wave arrival time error (not used)
   - Column 7: F-P time (not used)
   - Column 8: Maximum amplitude (not used)
   - Column 9: Station latitude (degrees)
   - Column 10: Station longitude (degrees)
   - Column 11: Station elevation (not used)
   - Column 12: P-wave station correction (if any, seconds, added to observed travel time)
   - Column 13: S-wave station correction (if any, but not used)

   *Output file format:*
   ```
   999 999 999 999 999 999 [longitude] [latitude] [depth] 999
   dummy 0. [latitude_error] [longitude_error] [depth_error]
   [cov_xx] [cov_xy] [cov_xz] [cov_yy] [cov_yz] [cov_zz]
   ```
   where some columns are not used by the code, filled with 999 or dummy.

#### Log File

For each processed event, a log file is generated in the specified `outputdir`:
- `log.XXXXXXXX.txt`: Contains detailed processing logs including:
  - Event ID
  - IO type
  - Initial hypocenter location defined by the program
  - Loss values during optimization (if conducted)
  - Processing time
  - Success/failure status

#### Example Usage

1. Process a single event:
```bash
hyponetn_run --src_s 0 --src_e 0 --eventdir input_events --outputdir results
```

2. Process a range of events:
```bash
hyponetn_run --src_s 0 --src_e 9 --eventdir input_events --outputdir results
```

OpenMP threading is enabled in the code. The number of threads can be set by the environment variable `OMP_NUM_THREADS`.


### As a Python Module

You can also import and use the models for travel time inference in your Python-based code for hypocenter determination. Here's how to use the package:

```python
from HypoNet_Nankai.HypoNet import HypoNet
import torch

# Initialize the model (don't change the inputdir and device)
model = HypoNet.model(inputdir="./input_p/", device=0)

# Example: Calculate travel times for a set of stations. Elevation is automatically set to be on the Earth's surface.
station_lon = torch.tensor([134.0, 135.0])
station_lat = torch.tensor([33.0, 33.5])

# Example hypocenter location
hypocenter_lon = torch.tensor([134.5, 134.5])
hypocenter_lat = torch.tensor([33.0, 33.0])
hypocenter_elev = torch.tensor([-10.0, -10.0]) # in km

# Calculate travel times
travel_times, errcodelist = model.inference_torch(hypocenter_lon, hypocenter_lat, hypocenter_elev, station_lon, station_lat)

# Output travel times are differentiable when input tensors have `requires_grad=True`
hypocenter_lon = torch.tensor([134.5, 134.5], requires_grad=True)
hypocenter_lat = torch.tensor([33.0, 33.0], requires_grad=True)
hypocenter_elev = torch.tensor([-10.0, -10.0], requires_grad=True) 
travel_times, errcodelist = model.inference_torch(hypocenter_lon, hypocenter_lat, hypocenter_elev, station_lon, station_lat)

YOUR_TT_DATA = torch.tensor([15., 51.]) # Your example travel time data
loss=((YOUR_TT_DATA-travel_times)**2).sum()
loss.backward()
print(hypocenter_lon.grad, hypocenter_lat.grad, hypocenter_elev.grad)


# errcodelist returns a list of error codes (see the definition below)
print(errcodelist) # output [0, 0] for the previous setting

station_lon = torch.tensor([134.0, 135.0])
station_lat = torch.tensor([33.5, 33.5])
hypocenter_lon = torch.tensor([134.5, 134.5])
hypocenter_lat = torch.tensor([33.0, 30.0])
hypocenter_elev = torch.tensor([-10.0, -10.0]) # in km

travel_times, errcodelist = model.inference_torch(hypocenter_lon, hypocenter_lat, hypocenter_elev, station_lon, station_lat)
print(errcodelist) # output: [1, 1] for the new setting

```

#### Key Methods

- `inference_torch(hypocenter_lon, hypocenter_lat, hypocenter_elev, station_lon, station_lat)`: Calculate P-wave travel times using PyTorch tensors for given stations and hypocenter location. Returns travel times and error codes.

#### Input Parameters

- `hypocenter_lon`: PyTorch tensor of shape (n_events,) containing hypocenter longitudes
- `hypocenter_lat`: PyTorch tensor of shape (n_events,) containing hypocenter latitudes
- `hypocenter_elev`: PyTorch tensor of shape (n_events,) containing hypocenter elevations in km
- `station_lon`: PyTorch tensor of shape (n_stations,) containing station longitudes
- `station_lat`: PyTorch tensor of shape (n_stations,) containing station latitudes

#### Output

- `inference_torch()` returns a tuple containing:
  - `travel_times`: PyTorch tensor of shape (n_events, n_stations) containing calculated travel times
  - `errcodelist`: List of error codes for each calculation. 0: both source and receiver within domain, 1: source or receiver outside domain (failed). Note travel time is returned even if errcode is 1.
- The output tensors are differentiable when input tensors have `requires_grad=True`, allowing for gradient-based optimization

## Project Structure

```
HypoNet_Nankai/
├── HypoNet/           # Base model implementation
├── HypoNet_DeeperPINN/# Deeper PINN model implementation
├── utils/             # Utility functions
├── input/            # General input data
├── input_p/          # P-wave specific input data
├── input_s/          # S-wave specific input data
└── main.py           # Main execution script
example/             # Example data and scripts for a numerical experiment presented in Agata+2025
```

## Author

- Ryoichiro Agata (agatar@jamstec.go.jp)

## License

MIT License

Copyright (c) 2025 Ryoichiro Agata

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

## Citation policy

If you use this code in your study, please cite the following paper in the acknowledgement section as below:

"HypoNet Nankai, a hypocenter determination tool developed by Agata et al. (2025) that includes modified bathymetry and geoid data from GEBCO Bathymetric Compilation Group 2023 (2023) and the Earth Gravitational Model 2008 (Pavlis et al. 2012), respectively, was used."

Reference:

Agata, R., Baba, S., Nakanishi, A., Nakamura, Y.: HypoNet Nankai: A rapid hypocenter determination tool for the Nankai Trough subduction zone using physics-informed neural network. Seismological Research Letters, accepted.

GEBCO Bathymetric Compilation Group 2023: The GEBCO_2023 Grid - a continuous terrain model of the global oceans and land. NERC EDS British Oceanographic Data Centre NOC (2023). doi:10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b

Pavlis, N.K., Holmes, S.A., Kenyon, S.C., Factor, J.K.: The development and evaluation of the Earth Gravitational Model 2008 (EGM2008). Journal of geophysical research: solid earth 117(B4) (2012). doi: 10.1029/2011JB008916


