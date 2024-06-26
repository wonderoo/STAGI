import numpy as np
from sklearn.svm import OneClassSVM, SVC

from utils.utils import ClassificationMetrices


def combine_datas(config):
    walk_path = config['file_path']
    import os
    res = []
    for root, nowdir, nowfiles in os.walk(walk_path):
        for file in nowfiles:
            if file != "flow_matrix.npy":
                res.append(root + "/" + file)

    datas = []
    for item in res:
        now = np.load(item)
        datas.append(now)

    datas = np.array(datas)
    datas = datas.transpose()
    print(datas.shape)
    return datas


class modelsSVM:
    def __init__(self, config):
        datas = np.load(config['matrix_path'])
        datas = datas.reshape(-1, datas.shape[2])
        self.feature = datas.transpose()
        print(self.feature.shape)
        self.feature = np.concatenate([self.feature, combine_datas(config)], axis=1)
        # print(self.feature.shape)

    def run(self, test_files):
        metrices = ClassificationMetrices()
        from tqdm import tqdm
        print("+1")
        cnt = 0
        for test_tuple in tqdm(test_files):
            sample_pos_list, sample_neg_list, positive_list, negative_list = test_tuple
            for i in range(len(sample_pos_list)):
                acc_pos = [0, 0]
                acc_neg = [0, 0]
                sample_pos = sample_pos_list[i].cpu().numpy()
                sample_neg = sample_neg_list[i].cpu().numpy()
                positive = positive_list[i].cpu().numpy()
                negative = negative_list[i].cpu().numpy()
                model = SVC(random_state=1234, kernel="rbf")

                input = []
                label = []
                for j in range(len(sample_pos)):
                    if sample_pos[j] == 1:
                        input.append(self.feature[j])
                        label.append(1)
                    if sample_neg[j] == 1:
                        input.append(self.feature[j])
                        label.append(-1)

                model.fit(input, label)

                for j in range(len(positive)):
                    if positive[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == 1:
                            acc_pos[0] += 1
                        acc_pos[1] += 1

                for j in range(len(negative)):
                    if negative[j] == 1:
                        now = self.feature[j].reshape(1, -1)
                        if model.predict(now) == -1:
                            acc_neg[0] += 1
                        acc_neg[1] += 1
                metrices.update((acc_pos, acc_neg))
            cnt += 1
            if cnt % 100 == 0:
                print(metrices.output("test", cnt))
        metrices = metrices.output("test", 0)
