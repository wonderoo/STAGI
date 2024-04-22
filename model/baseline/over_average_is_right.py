import torch
from utils.utils import load_files
import numpy as np


def result_matrix(test_matrix, train_matrix):
    matrix = torch.zeros(len(test_matrix), 4)

    matrix[:, 0] = torch.sum((test_matrix == 0) & (train_matrix == 0), dim=1)
    matrix[:, 1] = torch.sum((test_matrix == 1) & (train_matrix == 1), dim=1)
    matrix[:, 2] = torch.sum((test_matrix == 1) & (train_matrix == 0), dim=1)
    matrix[:, 3] = torch.sum((test_matrix == 0) & (train_matrix == 1), dim=1)

    sum_acc = (matrix[:, 0] + matrix[:, 1]).sum()
    sum_pos = (matrix[:, 1] + matrix[:, 2]).sum()
    sum_neg = (matrix[:, 0] + matrix[:, 3]).sum()
    acc_pos = matrix[:, 1].sum()
    acc_neg = matrix[:, 0].sum()
    print(matrix)
    return sum_acc / test_matrix.numel(), acc_pos / sum_pos, acc_neg / sum_neg


def over_average_is_right(train_loader, test_loader, val_loader, files):
    # 规格：类别数*区域数 值是区域某类别的数量 例如北京是N*1020
    # train_data = torch.tensor([[1, 0, 1], [2, 2, 0], [1, 0, 0], [1, 0, 1]])
    # test_data = torch.tensor([[1, 0, 1], [0, 0, 2]])
    # val_data = torch.tensor([[1, 0, 0], [0, 1, 0]])
    # train_samples, train_positives, train_negatives = [], [], []
    # for batch in train_loader:
    #     batch_samples, batch_positives, batch_negatives = batch
    #     train_samples.append(batch_samples)
    #     train_positives.append(batch_positives)
    #     train_negatives.append(batch_negatives)
    # train_data = torch.cat(train_samples, dim=0)
    #
    # test_samples, test_positives, test_negatives = [], [], []
    # for batch in test_loader:
    #     batch_samples, batch_positives, batch_negatives = batch
    #     test_samples.append(batch_samples)
    #     test_positives.append(batch_positives)
    #     test_negatives.append(batch_negatives)
    # test_data = torch.cat(test_samples, dim=0)
    #
    # val_samples, val_positives, val_negatives = [], [], []
    # for batch in val_loader:
    #     batch_samples, batch_positives, batch_negatives = batch
    #     val_samples.append(batch_samples)
    #     val_positives.append(batch_positives)
    #     val_negatives.append(batch_negatives)
    # val_data = torch.cat(val_samples, dim=0)
    test_acc_m, val_acc_m = 0,0
    NUM = 10
    for i in range(NUM):
        np.random.shuffle(files)
        rate_train, rate_val, rate_test = 0.7, 0.15, 0.15

        train_data = torch.tensor(files[:int(len(files) * rate_train)])
        val_data = torch.tensor(files[int(len(files) * rate_train):int(len(files) * (rate_train + rate_val))])
        test_data = torch.tensor(files[int(len(files) * (rate_train + rate_val)):])
        print(train_data)

        total_type_num = len(train_data) + len(test_data) + len(val_data)
        print(total_type_num)
        threshold = torch.tensor(total_type_num // 2)

        binary_region = (train_data > 0).int()

        region_weight = torch.sum(binary_region, dim=0, keepdim=True)
        print('region_w', region_weight.shape)
        print(region_weight)
        binary_region_weight = (region_weight >= threshold).int()

        print('binary_weight', binary_region_weight)

        binary_test_data = (test_data > 0).int()
        binary_val_data = (val_data > 0).int()
        print('binary_test_data\n', binary_test_data)

        # matrix 矩阵含义 N*4
        # 负样本 T--|正样本 T--|正样本 F--|负样本 F--|

        test_acc_m += result_matrix(binary_test_data, binary_region_weight)[0]
        val_acc_m += result_matrix(binary_val_data, binary_region_weight)[0]
    print('test_result', test_acc_m / NUM)
    print('val-result', val_acc_m / NUM)
    return test_acc_m, val_acc_m
