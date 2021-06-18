import os
import random
import time

import numpy as np
import pyhocon
import torch
import pandas as pds


def read_conf(path, name):
    if os.path.exists(path):
        conf = pyhocon.ConfigFactory.parse_file(path)[name]
    else:
        conf = None
        print('Unrecognized language')
        exit(0)
    print('Configuration: {}'.format(name))
    return {'name': name, 'conf': conf}


def get_device(gpu_id):
    gpu = 'cuda:' + str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device(gpu)
        print('Running on GPU: ' + str(gpu_id))
    else:
        device = torch.device('cpu')
        torch.set_num_threads(6)
        print('Running on CPU')
    return device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            torch.cuda.manual_seed(seed)
        elif torch.cuda.device_count() > 1:
            torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def get_timestamp():
    return time.strftime('%y-%m-%d-%H:%M:%S', time.localtime(time.time()))


def split_avg(li, n):
    splitted = []
    li_len = len(li)
    left = li_len % n
    if left != 0:
        li = li[:li_len - left]
    li_len = len(li)
    size = int(li_len / n)
    for i in range(0, li_len, size):
        splitted.append(li[i:i + size])
    return splitted


def match(li1, li2):
    assert len(li1) == len(li2)
    match_num = 0
    for i in range(len(li1)):
        if li1[i] == li2[i]:
            match_num += 1
    return match_num


def orthonormal_initializer(shape):
    M1 = np.random.randn(shape[0], shape[0]).astype(np.float32)
    M2 = np.random.randn(shape[1], shape[1]).astype(np.float32)
    Q1, R1 = np.linalg.qr(M1)
    Q2, R2 = np.linalg.qr(M2)
    Q1 = Q1 * np.sign(np.diag(R1))
    Q2 = Q2 * np.sign(np.diag(R2))
    n_min = min(shape[0], shape[1])
    params = np.dot(Q1[:, :n_min], Q2[:n_min, :])

    return params


def block_orthonormal_initializer(size_0, size_1, block_num):
    assert (size_1 % block_num) == 0
    block_size = int(size_1 / block_num)
    block_sizes = [block_size] * block_num
    params = np.concatenate([orthonormal_initializer([size_0, b]) for b in block_sizes], 1)

    return params


def square_error(li1, li2):
    assert len(li1) == len(li2)
    sum_suqare_error = 0.0
    for i in range(len(li1)):
        sum_suqare_error += (li1[i] - li2[i]) ** 2
    return sum_suqare_error


def save_as_csv(col_datas, col_names, path):
    assert len(col_names) == len(col_datas)
    to_csv = {col_name: col_datas[i] for i, col_name in enumerate(col_names)}
    data = pds.DataFrame(to_csv)
    data.to_csv(path, index=False, encoding='gb2312-80')
