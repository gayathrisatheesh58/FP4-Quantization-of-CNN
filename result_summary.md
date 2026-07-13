# Experimental Results

## Dataset
MNIST

## Model
CNN

## Parameters
206,922

## Quantization
FP4-E2M1

## Method
Block-wise quantization

## Results

| Model | Accuracy |
|------|------|
| FP32 | 98.80% |
| FP4 | 98.45% |

Accuracy drop:
0.35%

## Storage

FP32:
32 bits/parameter

FP4:
4 bits/parameter

Theoretical compression:
8×

Practical compression:
depends on scale overhead.
