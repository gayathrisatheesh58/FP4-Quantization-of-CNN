# FP4 Quantization of CNN for Efficient Edge AI Inference

## Overview

Deep Neural Networks (DNNs) provide excellent accuracy but require large memory and computational resources because model parameters are usually stored using high precision formats such as FP32 (32-bit Floating Point).

Low-bit quantization reduces the number of bits used to represent model parameters, reducing memory requirements and improving the efficiency of AI inference on resource-constrained edge devices.

This project implements an FP4 (4-bit Floating Point) quantization approach on a Convolutional Neural Network (CNN) trained for MNIST handwritten digit classification.

The objective of this project is to understand:

- Floating-point quantization
- FP4 numerical representation
- Scaling techniques
- Block-wise quantization
- Accuracy preservation after precision reduction
- Memory reduction possibilities for efficient AI hardware


---

# Project Workflow

The complete implementation follows these stages:

Stage 1:
FP32 CNN model training

↓

Stage 2:
Analyze trained model parameters and weight distribution

↓

Stage 3:
Design FP4 quantization method

↓

Stage 4:
Implement scaling-based FP4 quantization

↓

Stage 5:
Implement block-wise FP4 quantization

↓

Stage 6:
Generate FP4 quantized CNN model

↓

Stage 7:
Evaluate accuracy degradation

↓

Stage 8:
Analyze quantization error and memory reduction


---

# Motivation

Modern AI models are becoming increasingly larger, requiring:

- High memory capacity
- High computational power
- Large energy consumption

For edge AI applications, these requirements are challenging because edge devices have:

- Limited memory
- Limited processing capability
- Strict power constraints

Quantization provides a solution by reducing numerical precision.

Traditional representation:

FP32:

- 32 bits per value

FP4:

- 4 bits per value


Theoretical weight compression:

32 bits / 4 bits = 8× reduction


---

# Dataset

Dataset:

MNIST Handwritten Digit Dataset


Properties:

- Number of classes: 10
- Image size: 28 × 28
- Image type: Grayscale
- Training samples: 60,000
- Testing samples: 10,000


The dataset is automatically downloaded using PyTorch and does not require manual storage.


---

# CNN Model Architecture

A Convolutional Neural Network is used as the baseline model.

Architecture:

Input Image

↓

Convolution Layer 1

↓

Max Pooling

↓

Convolution Layer 2

↓

Max Pooling

↓

Fully Connected Layer 1

↓

Fully Connected Layer 2

↓

Output Classification


Total parameters:

206,922


The FP32 model is trained first and used as the baseline for comparison.


---

# FP4 Quantization Methodology

The quantization process is performed in multiple steps.

## Step 1: FP32 Training

The CNN is trained using normal FP32 precision.

Each weight is represented using:

32 bits


Example:

FP32 weight:

0.463579


---

## Step 2: FP4 Representation

The FP32 weights are mapped to a smaller FP4 numerical space.

FP4 uses:

- 4 bits per value
- Reduced representation levels
- Lower memory requirement


---

## Step 3: Scaling-Based Quantization

Direct FP4 conversion causes large precision loss because neural network weights usually have small values.

Therefore, scaling is introduced.

Process:

Weight values

↓

Calculate scaling factor

↓

Normalize weights

↓

Convert normalized values to FP4

↓

Reconstruct original range


This improves numerical precision.


---

## Step 4: Block-wise FP4 Quantization

Instead of using one scaling factor for the complete layer, weights are divided into smaller blocks.

Implementation:

Block size:

32 weights


Process:

Layer weights

↓

Divide into blocks of 32 values

↓

Calculate individual scale for each block

↓

Perform FP4 quantization

↓

Store quantized values with scaling information


Block-wise scaling provides better accuracy because each block has its own dynamic range.


---

# Experimental Results

## Accuracy Comparison

| Model | Precision | Accuracy |
|------|-----------|----------|
| CNN Baseline | FP32 | 98.80% |
| Quantized CNN | FP4 | 98.45% |


Accuracy degradation:

0.35%


The results demonstrate that FP4 quantization can significantly reduce numerical precision while maintaining classification performance.


---

# Quantization Error Analysis

Mean Absolute Error (MAE) after block-wise FP4 quantization:


| Layer | MAE |
|------|------|
| conv1.weight | 0.03024 |
| conv2.weight | 0.01614 |
| fc1.weight | 0.00711 |
| fc2.weight | 0.01429 |


Observations:

- Convolution layers show higher error due to smaller parameter count.
- Large fully connected layers benefit more from block-wise scaling.
- Proper scaling is essential for maintaining accuracy at low precision.


---

# Memory Analysis

FP32 Model:

Each weight:

32 bits


FP4 Quantized Model:

Each weight:

4 bits

Additional storage:

Block scaling factors


Theoretical compression:

FP32 / FP4

= 32 / 4

= 8× reduction


Practical compression depends on additional scale storage and hardware implementation.


---

# Important Observation

A direct FP4 conversion without scaling results in high quantization error.

The progression observed in this project:

Direct FP4 quantization

↓

High error



Layer-wise scaling

↓

Reduced error





Block-wise scaling

↓

Further reduction in error


This demonstrates why modern low-bit AI models use advanced quantization techniques.


---


---

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Google Colab
- Jupyter Notebook


---

# Future Work

This project serves as a foundation for exploring extremely low precision AI hardware acceleration.

Future extensions:

- FP2 quantization implementation
- Comparison between FP4 and FP2 inference
- Custom floating-point arithmetic hardware
- FPGA accelerator design
- Energy-efficient AI inference architectures


---

# Relation to FP2 Research

This FP4 implementation acts as a foundation for future FP2-based AI acceleration.

The workflow learned here:

FP32 Model

↓

Low precision representation

↓

Quantization algorithm

↓

Accuracy evaluation

↓

Hardware optimization


will be extended to FP2 formats for edge AI inference.


---

# References

1. LLM-FP4: Low Precision Floating Point Quantization for Large Language Models

2. FP2: A 2-bit Floating-Point Format for Edge-AI Inference and Fine-Tuning


---

# Author

Gayathri Satheesh

M.Tech Electronics

AI Hardware Acceleration and Low Precision Computing

