# FP4 Quantization of CNN for Efficient Edge AI Inference

## Overview

Deep learning models require large memory and computational resources, which creates challenges for deployment on edge devices with limited power and hardware capabilities.

Quantization is a technique used to reduce model size and computational cost by representing model parameters with lower precision numbers.

This project implements **FP4 (4-bit Floating Point) quantization on a Convolutional Neural Network (CNN)** trained on the MNIST handwritten digit classification dataset.

The objective is to study how reducing weight precision from **FP32 (32-bit floating point)** to **FP4 (4-bit floating point)** affects:

- Model accuracy
- Weight representation
- Quantization error
- Memory requirements

This project serves as a foundation for exploring ultra-low precision formats such as **FP2 quantization for edge AI accelerators**.

---

# Project Workflow
