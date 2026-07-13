import torch
import numpy as np
import copy


# FP4-E2M1 representation

fp4_values = [
    -1.75,
    -1.5,
    -1.0,
    -0.75,
    -0.5,
    -0.25,
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    1.75
]


def quantize_fp4(value):

    closest = min(
        fp4_values,
        key=lambda x: abs(x-value)
    )

    return closest



def quantize_tensor_fp4_blockwise(
        tensor,
        block_size=32):


    tensor_flat = (
        tensor.detach()
        .cpu()
        .flatten()
    )


    quantized_flat = torch.zeros_like(
        tensor_flat
    )


    scales = []


    for i in range(
        0,
        len(tensor_flat),
        block_size
    ):

        block = tensor_flat[i:i+block_size]


        max_value = torch.max(
            torch.abs(block)
        )


        if max_value == 0:

            scale = 1.0

        else:

            scale = max_value / 1.75


        scales.append(scale)


        normalized_block = (
            block / scale
        )


        quantized_block = torch.zeros_like(
            normalized_block
        )


        for index, value in enumerate(normalized_block):

            quantized_block[index] = quantize_fp4(
                value.item()
            )


        reconstructed_block = (
            quantized_block * scale
        )


        quantized_flat[i:i+block_size] = reconstructed_block


    return (
        quantized_flat.reshape(tensor.shape),
        scales
    )



def quantize_model_fp4(
        model,
        block_size=32):


    model_fp4 = copy.deepcopy(model)

    scales_dict = {}


    for name, param in model_fp4.named_parameters():


        if "weight" in name:


            quantized_weight, scales = quantize_tensor_fp4_blockwise(
                param.data,
                block_size
            )


            param.data = quantized_weight


            scales_dict[name] = scales


    return model_fp4, scales_dict
