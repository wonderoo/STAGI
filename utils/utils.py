import json
import numpy as np
import torch

from data.myDataset import myDataset

from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score

import logging
import os


def load_config(dataset):
    with open("./data/config.json", 'r', encoding='utf-8') as f:
        config = json.load(f)

    config = config[dataset]

    return config


def load_files(walk_path):
    print(walk_path)
    import os
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            res.append(root + "/" + file)

    datas = []
    for item in res:
        now = np.load(item)
        datas.append(now)

    return datas


def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logging(args):
    logging.basicConfig(level=logging.INFO)
    log_dir = os.path.join("./log", args.data)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'{args.model_name}_{args.seed}.log')
    file_handler = logging.FileHandler(log_file)
    logging.getLogger().addHandler(file_handler)
    logging.info(args)


def output(now):
    # print(now)
    logging.info(now)


# sample - positive - negative
def build_data(data, files, bsz, gpu, train=1, regression=1):
    res = myDataset(gpu, train, regression)

    for item in data:
        # iterater rate from 0.2 to 1
        for rate in range(5, 95, 5):
            # chosse the top k the largest value from the numpy array and return the boolean array
            anchor_value = np.percentile(item, 100 - rate)

            if anchor_value == 0:
                continue

            if len(np.where(item >= anchor_value)[0]) <= 1:
                continue

            if train == 1:
                res.append((item, anchor_value))

            else:
                for j in range(10):
                    positive = np.where(item >= anchor_value)[0]
                    np.random.shuffle(positive)
                    sample = positive[:int((len(positive) * 0.7) // 1)]
                    positive = positive[int((len(positive) * 0.7) // 1):]

                    negtive = np.where(item < anchor_value)[0]
                    np.random.shuffle(negtive)
                    negtive = negtive[:min(len(negtive), len(positive))]

                    grid_num = len(item)
                    zero_sample = torch.zeros(grid_num)
                    zero_sample[sample] = 1
                    zero_postive = torch.zeros(grid_num)
                    zero_postive[positive] = 1
                    zero_negtive = torch.zeros(grid_num)
                    zero_negtive[negtive] = 1

                    res.append((item, zero_sample, zero_postive, zero_negtive))

    return torch.utils.data.DataLoader(res, batch_size=bsz,
                                       shuffle=True)


def build_train_data(files, batch_size, gpu, regression):
    print(len(files))
    rate_train, rate_val, rate_test = 0.7, 0.15, 0.15

    train_data = files[:int(len(files) * rate_train)]
    val_data = files[int(len(files) * rate_train):int(len(files) * (rate_train + rate_val))]
    test_data = files[int(len(files) * (rate_train + rate_val)):]

    train_set = build_data(train_data, files, batch_size, gpu, train=1, regression=regression)
    val_set = build_data(val_data, files, batch_size, gpu, train=0, regression=regression)
    test_set = build_data(test_data, files, batch_size, gpu, train=0, regression=regression)

    return train_set, val_set, test_set


def evaluation(sum_pos, sum_neg):
    # sum pos is in the format of (right, total), sum neg is in the format of (right, total)
    # please calculate the roc_auc, f1, accuracy, precision, recall using sklearn

    truth = [1] * sum_pos[1] + [0] * sum_neg[1]
    prediction = [1] * sum_pos[0] + [0] * (sum_pos[1] - sum_pos[0]) + [0] * sum_neg[0] + [1] * (sum_neg[1] - sum_neg[0])

    roc_auc = roc_auc_score(truth, prediction)
    f1 = f1_score(truth, prediction)
    accuracy = accuracy_score(truth, prediction)
    precision = precision_score(truth, prediction)
    recall = recall_score(truth, prediction)

    return roc_auc, f1, accuracy, precision, recall


def evaluation_regression(data, tuple):
    # data is a list of numpy array, tuple is a list of tuple
    # please calculate the mae, mse, rmse using numpy
    # return the mae, mse, rmse

    truth = []
    prediction = []

    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if tuple[i][j][k][1] == 1 or tuple[i][j][k][2] == 1:
                    truth.append(data[i][j][k])
                    prediction.append(tuple[i][j][k][0])

    truth = np.array(truth)
    prediction = np.array(prediction)

    mae = np.mean(np.abs(truth - prediction))
    mse = np.mean((truth - prediction) ** 2)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(truth, prediction)[0, 1]

    return mae, mse, rmse, pcc


class RegressionMetrices:
    def __init__(self):
        self.data = []
        self.tuple = []

    def update(self, acc):
        item, input_tuple = acc
        self.data.append(item.detach().cpu().numpy())
        self.tuple.append(input_tuple)

    def output(self, phase, epoch):
        mae, mse, rmse, pcc = evaluation_regression(self.data, self.tuple)
        output(f"{phase}: Epoch: {epoch}, mae: {mae}, mse: {mse}, rmse: {rmse}, pcc: {pcc}")
        return mae, mse, rmse


class ClassificationMetrices:
    def __init__(self):
        self.sum_pos = (0, 0)
        self.sum_neg = (0, 0)

    def update(self, acc):
        acc_pos, acc_neg = acc
        self.sum_pos = (self.sum_pos[0] + acc_pos[0], self.sum_pos[1] + acc_pos[1])
        self.sum_neg = (self.sum_neg[0] + acc_neg[0], self.sum_neg[1] + acc_neg[1])

    def output(self, phase, epoch):
        roc_auc, f1, accuracy, precision, recall = evaluation(self.sum_pos, self.sum_neg)
        output(
            f"{phase}: Epoch: {epoch}, Acc Pos: {self.sum_pos[0], self.sum_pos[1]}, Acc Neg: {self.sum_neg[0], self.sum_neg[1]}")
        output(
            f"roc_auc: {roc_auc}, f1: {f1}, accuracy: {accuracy}, precision: {precision}, recall: {recall}")

        return roc_auc, f1, accuracy, precision, recall