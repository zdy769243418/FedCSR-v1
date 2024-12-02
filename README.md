# FedCSR: A Federated Framework for Multi-Platform Cross-Domain Sequential Recommendation with Dual Contrastive Learning

## 1. Introduction
This repository contains the code for the paper "FedCSR: A Federated Framework for Multi-Platform Cross-Domain Sequential Recommendation with Dual Contrastive Learning" which has been accepted by Coling 2025. FedCSR is a novel federated cross-domain sequential recommendation framework that addresses the challenges of data heterogeneity and privacy in multi-platform scenarios. It eliminates the need for user alignment between platforms and utilizes dual contrastive learning mechanisms (Model Contrastive Learning - MCL and Sequence Contrastive Learning - SCL) to enhance recommendation performance.

## 2. Requirements
  Python 3.6
  
  PyTorch 1.11.1+cu11
## 3. Datasets
We use public datasets from Amazon to simulate federated CSR scenarios. The datasets are divided into multiple cross-domain datasets, such as "Food—Kitchen" and "Movie-Book". Each user in these datasets has interaction sequences spanning different domains.

## 4. Model Architecture
Base Representation Encoder: Utilizes an attentional graph neural network (GAEncoder) to extract sequential representations for each domain. It encodes item-item matrices based on the temporal order of item occurrences in the sequences.
Model Contrastive Learning (MCL): Creates contrastive signals between local and global user representations to reduce the information gap and mitigate model drift between platforms. It treats local and global representations of the same domain as positive samples and other domain's global sequential representations as negative samples.
Sequence Contrastive Learning (SCL): Employs a cross-domain sequence augmentation technique (using replace and reorder operations) to boost diversity and balance in inter-domain sequences. It then aligns original and augmented representations to improve the quality of sequential representation.

## 5. How to Use
python train.py

## ６. Citation
If you use our code or find our work helpful, please cite our paper:
FedCSR: A Federated Framework for Multi-Platform Cross-Domain Sequential Recommendation with Dual Contrastive Learning. Coling 2025.
