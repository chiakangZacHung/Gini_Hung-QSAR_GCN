# Gini_Hung-QSAR_GCN

## General info

This project is QSAR modeling without descriptors using graph convolutional neural networks:The case of mutagenicity prediction
The research is published in Springer:
<p align="center">
  <img height="250" src="IMG/img1.png"> <img height="250" src="IMG/img6.png">
</p>

Two models based on Graph Convolutional Neural Networks (GCN) are presented, with and without  Bayesian estimation of the prediction uncertainty, to study mutagenicty.

<p align="center">
  <img height="500" src="IMG/img2.jpg">
</p>

## Technologies
Project is created with:
* Python version: 3.7.5
* Tensorflow version: 1.13.1
* Rdkit version: 2019.09.1
	
## Setup
To install the dependencies:

```
  $ conda create -c conda-forge -n my-rdkit-env rdki
  $ conda activate my-rdkit-env
  pip install tensorflow
  pip install numpy
  ```
  
