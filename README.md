# BayesianDeepModelCalibration.pytorch
Code for reproducing some of the experiments of the ICML 2019 submission named Calibration of Deep Probabilistic Models with Decoupled Bayesian Neural Networks. Follow the instructions.

## Important

As PyTorch is a recent software, the experiments performed in this work have been done over two different releases, as the second one provide lots of advantages over the initial one. Thus, to ensure reproducibility we provide code and details on the models in separate folders.

This code is uploaded just to justify the results provided in the work, and not for a general purpose application. This means that, for instance, code is only prepared to run in GPU and python2.7

In the work we report more than 40 experiments. We uploaded some data files to google drive, but not all of them. Feel free to send an email to [elmaronias@gmail.com](elmaronias@gmail.com) requesting the logits for a particular experiment.

## Software and Hardware Requirements

To ensure reproducibility we provide the hardware and software used.

Software: pytorch version 0.3.1, python 2.7, cuda 8, ubuntu 16 (stable version) ; pytorch version 0.4.0, python2.7, cuda9.1, ubuntu 16 (stable version)

Hardware (GPU): Nvidia 1080 and Nvidia TitanXp. Intel based platforms

## Installation

To install pytorch (in a virtual enviroment) follow the next steps:

Pytorch 0.3.1:

	virtualenv -p python2.7 0.3.1_cuda8_pytorch
        source 0.3.1_pytorch/bin/activate
        pip install  http://download.pytorch.org/whl/cu80/torch-0.3.1-cp27-cp27m-linux_x86_64.whl


Pytorch 0.4.0
 
	virtualenv -p python2.7 0.4.0_cuda9.1_pytorch
        source 0.4.0_cuda9.1_pytorch/bin/activate
        pip install  https://download.pytorch.org/whl/cu91/torch-0.4.0-cp27-cp27m-linux_x86_64.whl

## Dataset

You can download the logits for some experiments from [here](https://drive.google.com/drive/folders/1uJ06kNMDRGcZ-6xjxHYtqwC6ea4eSGcl?usp=sharing). Place the downloaded data in a directory, as example, /tmp/data/

