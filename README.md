# Robust Barycenters of Persistence Diagrams

This repository provides the code used to retrieve the running time results in the following paper.

## Reference

If you plan to use this code to generate results for a scientific document, thanks for referencing the following publication:

"Robust Barycenters of Persistence Diagrasm"
Keanu Sisouk, Eloi Tanguy, Julie Delon and Julien Tierny
IEEE Transactions on Visualization and Computer Graphics, 2026.

[Paper](https://arxiv.org/abs/2509.14904)

# Requirements

To install the required packages:
```bash
pip install -r requirements.txt
```

# Get the result

The python files generates the running time reported in our time table.
Beware that the experiments were run using Pytorch on an NVIDA GPU, so associated drivers have to be installed. The specs are specified in the associated paper, but we remind that the GPU used is a NVIDIA RTX 260 Q-max (mobile) with 6 GB VRAM dedicated.

The first file `running_time_B_analytic.py` gives the running times when using the arithmetic mean:
```bash
python3 running_time_B_analytic.py
```

The second file `running_time_B.py` gives the running times when using Pytorch optimizer to compute the ground barycenter:
```bash
python3 running_time_B.py
```

# Credits

The core of this implementation comes from the following [repository](https://github.com/eloitanguy/ot_bar/blob/main/README.md) associated to the following [work](https://arxiv.org/abs/2407.13445).

