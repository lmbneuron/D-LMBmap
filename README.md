# D-LMBmap
[![DOI](https://zenodo.org/badge/DOI/10.21203/rs.3.rs-2188055/v1.svg)](https://doi.org/10.21203/rs.3.rs-2188055/v1)
![logo](./D-LMBmap.png)

- [Paper](https://www.nature.com/articles/s41592-023-01998-6)
- [Software](https://github.com/lmbneuron/D-LMBmap/releases)
- [Axon Segmentation](https://github.com/lmbneuron/D-LMBmap/tree/main/Axon Segmentation)
- [Brain Region Segmentation](https://github.com/lmbneuron/D-LMBmap/tree/main/Brain Region Segmentation)
- [Brain Style Transfer](https://github.com/lmbneuron/D-LMBmap/tree/main/Brain Style Transfer)
- [Whole Brain Registration](https://github.com/lmbneuron/D-LMBmap/tree/main/Whole Brain Registration)

## 🗼 Overview

D-LMBmap, an end-to-end package providing an integrated workflow containing three modules based on novel deep-learning algorithms for whole-brain connectivity mapping: axon segmentation, brain region segmentation and whole-brain registration. D-LMBmap is the first method that does not require manual annotation for axon segmentation, and achieves quantitative analysis of whole-brain projectome in a single workflow with superior accuracy for multiple cell types in all of the modalities tested.

Our test dataset are accessible [here](https://drive.google.com/drive/folders/1wQ-gvxpLPXUk6D-2jdbttq_lOw1nzBzz?usp=share_link).


![pipeline](./pipeline.png)
## 🌟 D-LMBmap software

[D-LMBmap V3.0](https://github.com/lmbneuron/D-LMBmap/releases)

D-LMBmap provides an open-source software with convenient high-level application programming interfaces (APIs) that can be used to build applications and extensions. The GUI provides functionality for the selection and computation of different pre-trained deep models. 
You can use D-LMBmap software by downloading our software from github->[release](https://github.com/lmbneuron/D-LMBmap/releases).

Please refer to the [tutorial](https://github.com/lmbneuron/D-LMBmap/tree/main/tutorial) before using the software. 

## 🌟 Axon Segmentation
![Axon segmentation](https://github.com/lmbneuron/D-LMBmap/blob/main/Axon%20Segmentation/Data%20Preparation%20and%20Axon%20Segmentation/data/axon%20segmentation.png)
We provide a whole-brain axon segmentation pipeline and model training method, which can be accessed [here](https://github.com/lmbneuron/D-LMBmap/tree/main/Axon%20Segmentation).

Axon segmentation models have been integrated into D-LMBmap software. The Axon segmentation function in the software can be used for cubes of any size, including whole-brain segmentation.

## 🌟 Brain Style Transfer & Brain Region Segmentation

![Brain region segmentation and Brain style Transfer](https://github.com/lmbneuron/D-LMBmap/blob/main/Brain%20Region%20Segmentation/Brain%20region%20segmentation%20and%20brain%20style%20transfer.png)

Brain region segmentation and brain style transfer methods mainly serve for whole brain registration. We also provide model training and prediction methods respectively, please refer to [Brain style transfer](https://github.com/lmbneuron/D-LMBmap/tree/main/Brain%20Style%20Transfer) and [Brain region segmentation](https://github.com/lmbneuron/D-LMBmap/tree/main/Brain%20Region%20Segmentation).

Some models have been integrated into D-LMBmap software.

## 🌟 Whole Brain Registration
![Architecture](https://github.com/lmbneuron/D-LMBmap/blob/main/Whole%20Brain%20Registration/imgs/architecture.png)
[Here](https://github.com/lmbneuron/D-LMBmap/tree/main/Whole%20Brain%20Registration) is the code for the multi-constraint and multi-scale facilitated whole-brain registration, axon density related heatmap generation.

## 🙋‍♀️ Feedback and Contact

If you encounter any issues, please feel free to contact us at zhongyuli@xjtu.edu.cn

## 🎫 License

This work is licensed under a Creative Commons Attribution 4.0 International License
