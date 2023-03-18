# Self-Modeling-Hopfield-Network

This repository contains Python code for running simulations of Self-Modeling Hopfield Networks, as described in the paper "Optimization in 'Self-Modeling' Complex Adaptive Systems" by Richard A. Watson, C. L. Buckley, and Rob Mills.

Overview

This repository includes two Python scripts:

m1_random.py: This script implements the self-modeling Hopfield network for random weights and reproduces the experiment reported in Figure 2 of the paper.
m1_modular_csp.py: This script implements the self-modeling Hopfield network to solve a modular weighted-2-max-sat problem and reproduces the experiment reported in Figure 3 of the paper.

In addition, the m1_100_trial.py script runs 100 independent trials of the m1_modular_csp.py experiment.

Requirements
The code was written and tested with Python 3.7.9, and requires the following libraries to be installed:

NumPy
Matplotlib

Usage

To run the experiments, simply run the corresponding Python scripts in a terminal or in an IDE such as Jupyter Notebook. The scripts will output plots similar to those shown in the paper.

The output of the scripts will be printed to the console and/or saved to files in the same directory as the scripts.

Acknowledgements
The code in this repository was created and is being updated by Nam Le-Hai as part of the project "Connectionist approaches to the evolutionary transitions in individuality" at the University of Southampton with Professor Richard Watson.

The code is intended for researchers and enthusiasts in the field of complex adaptive systems and provides a simple and easily understandable implementation of the self-modeling Hopfield network. The project is ongoing and may be updated in the future to include additional experiments or improvements to the existing code.

Citation
If you use this code for research purposes, please cite the following paper:

Watson, R. A., Buckley, C. L., & Mills, R. (2018). Optimization in "Self-Modeling" Complex Adaptive Systems. https://onlinelibrary.wiley.com/doi/abs/10.1002/cplx.20346