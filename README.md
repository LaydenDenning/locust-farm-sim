# Crop Monitoring Drone Farm Simulation

Simulation environment for evaluating the value of UAV-based crop monitoring
using PCSE and WOFOST.

## Objective

The project compares conventional crop scouting against UAV-assisted crop
monitoring using simulated crop ground truth.

Initial development focuses on generating spatially heterogeneous crop
conditions using WOFOST/PCSE.

## Simulation Architecture

1. WOFOST/PCSE generates crop ground truth
2. Farm is divided into spatial management zones
3. Crop stresses are introduced into selected zones
4. Conventional scouting is simulated
5. RGB/NIR UAV observations are simulated
6. Monitoring strategies are compared
7. Management and economic impacts are evaluated

## Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
