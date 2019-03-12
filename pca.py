# -*- coding: utf-8 -*
import numpy as np


def pca(train_label, test_label, k):
    test_matrix = test_label[:, :-1]
    train_matrix = train_label[:, :-1]

    # 中心化
    train_center = []
    for i in range(train_matrix.shape[0]):
        line_mean1 = np.mean(train_matrix[i])
        train_center.append(train_matrix[i] - line_mean1)
    train_center = np.array(train_center)

    # 协方差特征分解
    cov = np.cov(train_center.T)
    eig_val, eig_vec = np.linalg.eig(cov)
    #
    # eig_pairs = [(eig_val[i], eig_vec[:, i]) for i in range(len(eig_val))]
    # eig_pairs.sort(key=lambda arr: arr[0], reverse=True)

    # features = []
    # for i in range(k):
    #     features.append(eig_pairs[i][1])

    index_arr = np.argpartition(eig_val, -k)[-k:]
    features = eig_vec[index_arr]
    # 降维
    train_matrix_reduced = np.transpose(np.dot(features, train_matrix.T))
    test_matrix_reduced = np.transpose(np.dot(features, test_matrix.T))

    train_label_reduced = np.c_[train_matrix_reduced, train_label[:, -1]]
    test_label_reduced = np.c_[test_matrix_reduced, test_label[:, -1]]

    return train_label_reduced, test_label_reduced


if __name__ == '__main__':
    train_label = np.loadtxt('./two_datasets/sonar-train.txt', delimiter=',')
    test_label = np.loadtxt('./two_datasets/sonar-test.txt', delimiter=',')
    # train_label = np.loadtxt('./two_datasets/splice-train.txt', delimiter=',')
    # test_label = np.loadtxt('./two_datasets/splice-test.txt', delimiter=',')

    k_arr = [10, 20, 30]
    for k in k_arr:
        train_label_reduced, test_label_reduced = pca(train_label, test_label, k)
        from oneNN import oneNN
        acc = oneNN(train_label_reduced, test_label_reduced)
        print("dataset reduced to {!s}, pca test acc is {!s}".format(k, acc))
