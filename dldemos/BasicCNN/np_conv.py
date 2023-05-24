import numpy as np
import pytest
import torch


def conv2d(input: np.ndarray,
           weight: np.ndarray,
           stride: int,
           padding: int,
           dilation: int,
           groups: int,
           bias: np.ndarray = None) -> np.ndarray:
    """2D Convolution Implemented with NumPy.

    Args:
        input (np.ndarray): The input NumPy array of shape (H, W, C).
        weight (np.ndarray): The weight NumPy array of shape
            (C', F, F, C / groups).
        stride (int): Stride for convolution.
        padding (int): The count of zeros to pad on both sides.
        dilation (int): The space between kernel elements.
        groups (int): Split the input to groups.
        bias (np.ndarray | None): The bias NumPy array of shape (C').
            Default: None.

    Outputs:
        np.ndarray: The output NumPy array of shape (H', W', C')
    """
    h_i, w_i, c_i = input.shape
    c_o, f, f_2, c_k = weight.shape

    assert (f == f_2)
    assert (c_i % groups == 0)
    assert (c_o % groups == 0)
    assert (c_i // groups == c_k)
    if bias is not None:
        assert (bias.shape[0] == c_o)

    f_new = f + (f - 1) * (dilation - 1)
    weight_new = np.zeros((c_o, f_new, f_new, c_k), dtype=weight.dtype)
    for i_c_o in range(c_o):
        for i_c_k in range(c_k):
            for i_f in range(f):
                for j_f in range(f):
                    i_f_new = i_f * dilation
                    j_f_new = j_f * dilation
                    weight_new[i_c_o, i_f_new, j_f_new, i_c_k] = \
                        weight[i_c_o, i_f, j_f, i_c_k]

    input_pad = np.pad(input, [(padding, padding), (padding, padding), (0, 0)])

    def cal_new_sidelngth(sl, s, f, p):
        return (sl + 2 * p - f) // s + 1

    h_o = cal_new_sidelngth(h_i, stride, f_new, padding)
    w_o = cal_new_sidelngth(w_i, stride, f_new, padding)

    output = np.empty((h_o, w_o, c_o), dtype=input.dtype)

    c_o_per_group = c_o // groups

    for i_h in range(h_o):
        for i_w in range(w_o):
            for i_c in range(c_o):
                i_g = i_c // c_o_per_group
                h_lower = i_h * stride
                h_upper = i_h * stride + f_new
                w_lower = i_w * stride
                w_upper = i_w * stride + f_new
                c_lower = i_g * c_k
                c_upper = (i_g + 1) * c_k
                input_slice = input_pad[h_lower:h_upper, w_lower:w_upper,
                                        c_lower:c_upper]
                kernel_slice = weight_new[i_c]
                output[i_h, i_w, i_c] = np.sum(input_slice * kernel_slice)
                if bias:
                    output[i_h, i_w, i_c] += bias[i_c]
    return output


@pytest.mark.parametrize('c_i, c_o', [(3, 6), (2, 2)])
@pytest.mark.parametrize('kernel_size', [3, 5])
@pytest.mark.parametrize('stride', [1, 2])
@pytest.mark.parametrize('padding', [0, 1])
@pytest.mark.parametrize('dilation', [1, 2])
@pytest.mark.parametrize('groups', ['1', 'all'])
@pytest.mark.parametrize('bias', [False])
def test_conv(c_i: int, c_o: int, kernel_size: int, stride: int, padding: str,
              dilation: int, groups: str, bias: bool):
    if groups == '1':
        groups = 1
    elif groups == 'all':
        groups = c_i

    if bias:
        bias = np.random.randn(c_o)
        torch_bias = torch.from_numpy(bias)
    else:
        bias = None
        torch_bias = None

    input = np.random.randn(20, 20, c_i)
    weight = np.random.randn(c_o, kernel_size, kernel_size, c_i // groups)

    torch_input = torch.from_numpy(np.transpose(input, (2, 0, 1))).unsqueeze(0)
    torch_weight = torch.from_numpy(np.transpose(weight, (0, 3, 1, 2)))
    torch_output = torch.conv2d(torch_input, torch_weight, torch_bias, stride,
                                padding, dilation, groups).numpy()
    torch_output = np.transpose(torch_output.squeeze(0), (1, 2, 0))

    numpy_output = conv2d(input, weight, stride, padding, dilation, groups,
                          bias)

    assert np.allclose(torch_output, numpy_output)
