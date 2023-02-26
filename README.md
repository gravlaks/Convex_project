# Randomized Gauss Newton on Neural Networks

## Authors: 

Torstein Ørbeck Eliassen  
Adrien Lemercier

## Overview

An efficient Gauss Newton approach using randomized projection based on [1] is implemented. Improvements are made to mitigate overshooting. A fully modular implementation allows for testing the  algorithm on arbitrarily structured neural networks.

2 algorithms are used as a baseline for our approach: Algo 1 that does per-sample projection, algo 2 that does full jacobian projection. The images in this repository show the initial results from both optimization algorithms. For a more thourough presentation of the full batch optimization, refer to slides found in documentation (docs).

## Results: 


### Both algos: 

*Optimization method comparison*
![Optimization method comparison](https://github.com/gravlaks/Rand-Gauss-Newton-for-NN/blob/main/plots/Opti_method_comp.png)

### Algo 1: 
*Projection method comparison*
![Projection method comparison](https://github.com/gravlaks/Rand-Gauss-Newton-for-NN/blob/main/plots/Projection_method_comp.png)

### Algo 2: 10 digit classification

*Results for Convolutional Neural Net*
![Results for Convolutional Neural Net](https://github.com/gravlaks/Rand-Gauss-Newton-for-NN/blob/main/plots/Classifier10dim.png)

### Algo 2: With vs. without backtracking line search 

*Backtracking on 4 hidden layer Neural Network*
![alt text](https://github.com/gravlaks/Rand-Gauss-Newton-for-NN/blob/main/plots/Backtrackcomparison.png)

### Algo 2: Projection size comparison

*Comparing batch sizes for random sampling*
![Comparing batch sizes for random sampling](https://github.com/gravlaks/Rand-Gauss-Newton-for-NN/blob/main/plots/Batch_size_comp.png)


## References: 

[1] :Ergen, T.; Candes, E.; Pilanci, M. Random Projections for Learning Non-convex Models
