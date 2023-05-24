from typing import Dict, Tuple

import numpy as np
import pytest
import torch


def conv2d_forward(input: np.ndarray, weight: np.ndarray, bias: np.ndarray,
                   stride: int, padding: int) -> Dict[str, np.ndarray]:
    """2D Convolution Forward Implemented with NumPy.

    Args:
        input (np.ndarray): The input NumPy array of shape (H, W, C).
        weight (np.ndarray): The weight NumPy array of shape
            (C', F, F, C).
        bias (np.ndarray | None): The bias NumPy array of shape (C').
            Default: None.
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.

    Outputs:
        Dict[str, np.ndarray]: Cached data for backward prop.
    """
    h_i, w_i, c_i = input.shape
    c_o, f, f_2, c_k = weight.shape

    assert (f == f_2)
    assert (c_i == c_k)
    assert (bias.shape[0] == c_o)

    input_pad = np.pad(input, [(padding, padding), (padding, padding), (0, 0)])

    def cal_new_sidelngth(sl, s, f, p):
        return (sl + 2 * p - f) // s + 1

    h_o = cal_new_sidelngth(h_i, stride, f, padding)
    w_o = cal_new_sidelngth(w_i, stride, f, padding)

    output = np.empty((h_o, w_o, c_o), dtype=input.dtype)

    for i_h in range(h_o):
        for i_w in range(w_o):
            for i_c in range(c_o):
                h_lower = i_h * stride
                h_upper = i_h * stride + f
                w_lower = i_w * stride
                w_upper = i_w * stride + f
                input_slice = input_pad[h_lower:h_upper, w_lower:w_upper, :]
                kernel_slice = weight[i_c]
                output[i_h, i_w, i_c] = np.sum(input_slice * kernel_slice)
                output[i_h, i_w, i_c] += bias[i_c]

    cache = dict()
    cache['Z'] = output
    cache['W'] = weight
    cache['b'] = bias
    cache['A_prev'] = input
    return cache


def conv2d_backward(dZ: np.ndarray, cache: Dict[str, np.ndarray], stride: int,
                    padding: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2D Convolution Backward Implemented with NumPy.

    Args:
        dZ: (np.ndarray): The derivative of the output of conv.
        cache (Dict[str, np.ndarray]): Record output 'Z', weight 'W', bias 'b'
            and input 'A_prev' of forward function.
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.

    Outputs:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The derivative of W, b,
            A_prev.
    """
    W = cache['W']
    b = cache['b']
    A_prev = cache['A_prev']
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    dA_prev = np.zeros(A_prev.shape)

    _, _, c_i = A_prev.shape
    c_o, f, f_2, c_k = W.shape
    h_o, w_o, c_o_2 = dZ.shape

    assert (f == f_2)
    assert (c_i == c_k)
    assert (c_o == c_o_2)

    A_prev_pad = np.pad(A_prev, [(padding, padding), (padding, padding),
                                 (0, 0)])
    dA_prev_pad = np.pad(dA_prev, [(padding, padding), (padding, padding),
                                   (0, 0)])

    for i_h in range(h_o):
        for i_w in range(w_o):
            for i_c in range(c_o):
                h_lower = i_h * stride
                h_upper = i_h * stride + f
                w_lower = i_w * stride
                w_upper = i_w * stride + f

                input_slice = A_prev_pad[h_lower:h_upper, w_lower:w_upper, :]
                # forward
                # kernel_slice = W[i_c]
                # Z[i_h, i_w, i_c] = np.sum(input_slice * kernel_slice)
                # Z[i_h, i_w, i_c] += b[i_c]

                # backward
                dW[i_c] += input_slice * dZ[i_h, i_w, i_c]
                dA_prev_pad[h_lower:h_upper,
                            w_lower:w_upper, :] += W[i_c] * dZ[i_h, i_w, i_c]
                db[i_c] += dZ[i_h, i_w, i_c]

    if padding > 0:
        dA_prev = dA_prev_pad[padding:-padding, padding:-padding, :]
    else:
        dA_prev = dA_prev_pad
    return dW, db, dA_prev


@pytest.mark.parametrize('c_i, c_o', [(3, 6), (2, 2)])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
def test_conv(c_i: int, c_o: int, kernel_size: int, stride: int, padding: str):

    # Preprocess
    input = np.random.randn(20, 20, c_i)
    weight = np.random.randn(c_o, kernel_size, kernel_size, c_i)
    bias = np.random.randn(c_o)

    torch_input = torch.from_numpy(np.transpose(
        input, (2, 0, 1))).unsqueeze(0).requires_grad_()
    torch_weight = torch.from_numpy(np.transpose(
        weight, (0, 3, 1, 2))).requires_grad_()
    torch_bias = torch.from_numpy(bias).requires_grad_()

    # forward
    torch_output_tensor = torch.conv2d(torch_input, torch_weight, torch_bias,
                                       stride, padding)
    torch_output = np.transpose(
        torch_output_tensor.detach().numpy().squeeze(0), (1, 2, 0))

    cache = conv2d_forward(input, weight, bias, stride, padding)
    numpy_output = cache['Z']

    assert np.allclose(torch_output, numpy_output)

    # backward
    torch_sum = torch.sum(torch_output_tensor)
    torch_sum.backward()
    torch_dW = np.transpose(torch_weight.grad.numpy(), (0, 2, 3, 1))
    torch_db = torch_bias.grad.numpy()
    torch_dA_prev = np.transpose(torch_input.grad.numpy().squeeze(0),
                                 (1, 2, 0))

    dZ = np.ones(numpy_output.shape)
    dW, db, dA_prev = conv2d_backward(dZ, cache, stride, padding)

    assert np.allclose(dW, torch_dW)
    assert np.allclose(db, torch_db)
    assert np.allclose(dA_prev, torch_dA_prev)
