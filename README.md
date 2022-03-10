# Quantum Phase Estimation

This library is a framework for quantum phase estimation, a fundamental building block to many quantum algorithms.


## Authors
- [@hjaleta](https://www.github.com/hjaleta)
- [@riccardo205](https://www.github.com/riccardo205)
- [@cbelo](https://www.github.com/cbelo)

## Requirements
* qiskit
* numpy
* quantuminspire

## Modules

The modules in **QPE** are the following: 

### PhaseEstimator
This module contains the base class of all algorithms

### Unitary
This module contains the representations of the unitary operator we use in QPE

### Iterative
Contains stuff related to the Iterative (IPEA) method

### Kitaev
Contains stuff related to the Kitaev method

### QFT
Contains stuff related to the Quantum Fourier Transform (QFT) and the Approximate Quantum Fourier Transform (AQFT)

### Backend
This module contains functions that facilitate the connection to the various backends the algorithms run on

### ExperimentRunner
This module can help you run several experiments at once

### UnpackData
With this module you can unpack data from an old, saved experiment

### HelpFunctions
Some smaller, general functions that are needed in several different places
