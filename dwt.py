
import math
import numpy as np
import copy
import pywt

# LOW_DECOMP = [1/2, 1/2]

LOW_DECOMP = [0.7071067811865476, 0.7071067811865476]
HIGT_DECOMP = [0.7071067811865476, -0.7071067811865476]
# HIGT_DECOMP = [1/2, -1/2]

LOW_RECOMP = [0.7071067811865476, 0.7071067811865476]
HIGT_RECOMP = [-0.7071067811865476, 0.7071067811865476]


def haar_filter_group_decomp(data, model='periodic'):
    filter_len = len(LOW_RECOMP)
    low_filter = np.asarray(LOW_DECOMP)
    hiht_filer = np.asarray(HIGT_DECOMP)

    data = np.array(data)
    pad_len = filter_len - 1
    data_pad = np.pad(data, (pad_len, 0), mode='symmetric')
    data_len = data_pad.shape[0]

    low_data = np.zeros(shape=(data.shape))
    hiht_data = np.zeros(shape=(data.shape))
    for i in range(data_len - 1):
        low_data[i] = np.sum(data_pad[i:i + 2] * low_filter)
        hiht_data[i] = np.sum(data_pad[i:i + 2] * hiht_filer)

    higt_downsample = hiht_data[1::2]
    low_downsample = low_data[1::2]
    return (low_downsample, higt_downsample)


def haar_filter_group_recomp(*parm, model='periodic'):
    cA, cD = parm

    low_filter_inverse = np.asarray(LOW_RECOMP)
    high_filter_inverse = np.asarray(HIGT_RECOMP)

    low_data_upsample = np.zeros(shape=(2*cA.shape[0]+1))
    high_data_upsample = np.zeros(shape=(2* cD.shape[0]+1))
    low_data_upsample[1::2] = cA
    high_data_upsample[1::2] = cD
    low_recomposition = np.zeros(shape=(2*cA.shape[0]))
    high_recomposition = np.zeros(shape=(2*cA.shape[0]))

    for i in range(2*cA.shape[0]):

        low_recomposition[i] = sum(low_data_upsample[i:i+2]*low_filter_inverse)
        high_recomposition[i] = sum(high_data_upsample[i:i+2]*high_filter_inverse)
    data = low_recomposition+high_recomposition
    return data


#


def haar_single_recomp(*parm):
    assert len(parm) == 2, "provided {} don't match".format(parm)
    cA, cD = parm[0], parm[1]
    coeffs_len = cA.shape[0]
    data = np.zeros(shape=2 * coeffs_len, dtype=np.float32)
    for i in range(0, coeffs_len):
        data[2 * i] = cA[i] * HIGT_RECOMP[0] + cD[i] * HIGT_RECOMP[1]
        data[2 * i + 1] = cA[i] * LOW_RECOMP[0] + cD[i] * LOW_RECOMP[1]
    return data


def haar_mutilevel_recomp(*parm):
    assert len(parm) == 1, "provided {} don't match".format(parm)
    coeffs = parm[0]
    level = len(coeffs) - 1
    cA, cD = coeffs[0], coeffs[1]

    coeffs.pop(0)
    coeffs.pop(0)
    if isinstance(cA, np.ndarray):
        len_cA = cA.shape[0]
    else:
        raise AttributeError("only accept array_like {}".format(cA))
    for i in range(1, level + 1):
        data = np.zeros(shape=(len_cA * 2,), dtype=np.float32)
        for j in range(len_cA):
            data[2 * j] = cA[j] * LOW_DECOMP[0] + cD[j] * LOW_DECOMP[1]
            data[2 * j + 1] = cA[j] * HIGT_DECOMP[0] + cD[j] * HIGT_DECOMP[1]
        cA = data
        len_cA = data.shape[0]
        if not i == level:
            del data
            cD = coeffs[0]
            coeffs.pop(0)
        else:
            break

    return data



def haar_single_decomp(data):
    low = np.asarray(LOW_DECOMP, dtype=np.float32)
    hiht = np.asarray(HIGT_DECOMP, dtype=np.float32)
    data_tmp = data
    trans_len = int(len(data_tmp) / 2)
    cA = np.zeros(shape=(trans_len,), dtype=np.float32)
    cD = np.zeros(shape=(trans_len,), dtype=np.float32)

    for j in range(0, len(data_tmp), 2):
        # a = sum(data[j:j + 2] * low)
        cA[j // 2] = sum(data[j:j + 2:, ] * low).astype('float32')
        cD[j // 2] = sum(data[j:j + 2:, ] * hiht).astype('float32')

    return cA, cD


def haar_mutilevel_decomp(data, model='symmetric', level=1):
    low = np.asarray(LOW_DECOMP, dtype=np.float32)
    hiht = np.asarray(HIGT_DECOMP, dtype=np.float32)

    order_max = np.log2(len(data)).astype('int8')
    assert level <= order_max, "proviede data can't be decomposed {} times".format(level)

    data_tmp = data
    order_max = level
    coeff_list = []

    for i in range(order_max):
        if coeff_list:
            coeff_list.pop(0)

        trans_len = int(len(data_tmp) / 2)
        cA = np.zeros(shape=(trans_len,), dtype=np.float32)
        cD = np.zeros(shape=(trans_len,), dtype=np.float32)

        for j in range(0, len(data_tmp), 2):
            # a = sum(data[j:j + 2] * low)
            cA[j // 2] = sum(data_tmp[j:j + 2:, ] * low).astype('float32')
            cD[j // 2] = sum(data_tmp[j:j + 2:, ] * hiht).astype('float32')
        coeff_list.insert(0, cD)
        coeff_list.insert(0, cA)
        data_tmp = cA
    return coeff_list


def haar_decomposition(data, model='symmetric', level=1):
    if isinstance(data, list):
        data = np.asarray(data, dtype=np.float32)
    if isinstance(data, tuple):
        data = np.asarray(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
        data = data
    else:
        raise TypeError("only accept array list tuple input;")

    if data.ndim == 1:
        if level == 1:
            cA, cD = haar_single_decomp(x, model=model)
            return cA, cD
        else:
            coeffs = haar_mutilevel_decomp(data, model=model, level=level)
        return coeffs


def haar_recomposition(*parm):
    if len(parm) == 2:
        cA, cD = parm
        data = haar_single_decomp(cA, cD)
        return data

    elif len(parm) == 1:
        coeffs = parm[0]
        if len(coeffs) >= 3:
            if isinstance(coeffs[-1], np.ndarray):
                data = haar_mutilevel_recomp(coeffs)
                return data

            else:
                raise AttributeError("provided {} don't match wavelet coeffs ".format(coeffs))
        else:
            raise ValueError("provided {} don't match wavelet coeffs ".format(coeffs))


# def haar_2D_decomp(data, level):
#     assert isinstance(data, np.ndarray), "only accept array_like data"
#     assert data.ndim == 2, "only accept two dimensions data,got {} dimensions".format(data.ndim)
#     heiht, width = data.shape
#     assert heiht == width, "provieded the two dimensions of data is not equal"
#     new_data = np.zeros(shape=(heiht, width))
#
#     low = np.asarray(LOW_DECOMP)
#     low = np.tile(low, (heiht, 1))
#     hiht = np.asarray(HIGT_DECOMP)
#     hiht = np.tile(hiht, (heiht, 1))
#
#     half = heiht // 2
#     LL = np.zeros(shape=(heiht // 2, width // 2))
#     LH = np.zeros(shape=(heiht // 2, width // 2))
#     HL = np.zeros(shape=(heiht // 2, width // 2))
#     HH = np.zeros(shape=(heiht // 2, width // 2))
#
#
#     for j in range(0, heiht, 2):
#         a = data[:, j:j + 2] * low
#         b = np.sum(a, 1)
#         c = new_data[:, j // 2]
#         new_data[:, j // 2] = np.sum(data[:, j:j + 2:, ] * low, 1)
#         new_data[:, half + j // 2] = np.sum(data[:, j:j + 2:, ] * hiht, 1)
#
#     data = np.transpose(new_data, (1, 0))
#     for i in range(0, heiht, 2):
#         # a = sum(data[j:j + 2] * low)
#         new_data[:, i // 2] = np.sum(data[:, i:i + 2:, ] * low, 1)
#         new_data[:, half + i // 2] = np.sum(data[:, i:i + 2:, ] * hiht, 1)
#     LL[:, :] = new_data[:heiht // 2, :width // 2]
#     HL[:, :] = new_data[:heiht // 2, width // 2:]
#     LH[:, :] = new_data[heiht // 2:, :width // 2]
#     HH[:, :] = new_data[heiht // 2:, width // 2:]
#     coeffs = (LL, (LH, HL, HH))
#
#     return coeffs


# def haar_2D_recomposition(*coeffs, level):
#     low = np.asarray(LOW_DECOMP, dtype=np.float32)
#     hiht = np.asarray(HIGT_DECOMP, dtype=np.float32)
#     order_max = np.log2(len(data))
#     if level == 1:
#         order_max = level
#
#     data_tmp = data
#     cA = np.zeros(shape=(1, len(data) / 2), dtype=np.float32)
#     cD = np.zeros(shape=(1, len(data / 2)), dtype=np.float32)
#     for i in range(order_max):
#         for j in range(0, len(data), 2):
#             cA[j // 2] = data_tmp[j:j + 1] * low
#             cD[j // 2] = data_tmp[j:j + 1] * hiht
#         data_xtmp = cA
#     return cA, cD


# def haar_2D_decomposition(*coeffs, level):
#     low = np.asarray([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=np.float32)
#     hiht = np.asarray([1 / np.sqrt(2), -1 / np.sqrt(2)], dtype=np.float32)
#     order_max = np.log2(len(data))
#     if level == 1:
#         order_max = level
#
#     data_tmp = data
#     cA = np.zeros(shape=(1, len(data) / 2), dtype=np.float32)
#     cD = np.zeros(shape=(1, len(data / 2)), dtype=np.float32)
#     for i in range(order_max):
#         for j in range(0, len(data), 2):
#             cA[j // 2] = data_tmp[j:j + 1] * low
#             cD[j // 2] = data_tmp[j:j + 1] * hiht
#         data_xtmp = cA
#     return cA, cD

if __name__ == "__main__":
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    cA, cD = haar_filter_group_decomp(x)
    cA_expect, cD_expect = pywt.dwt(x, 'db1')
    print(cA, cD)
    print(cA_expect, cD_expect)
    data = haar_filter_group_recomp(cA,cD)
    print(data)
    # level = 1
    # if level >= 2:
    #     coeffs = haar_decomposition(x, level=level)
    #     print(coeffs)
    # elif level == 1:
    #     cA, cD = haar_decomposition(x)
    #     print(cA, cD)
    # else:
    #     raise ValueError("provieded {} value is not correctly".format(level))
