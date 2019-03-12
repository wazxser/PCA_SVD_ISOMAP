# -*- coding: utf-8 -*
import numpy as np


def svd(train_label, test_label, k):
    test_matrix = test_label[:, :-1]
    train_matrix = train_label[:, :-1]

    # SVD分解
    (U, S, VT) = np.linalg.svd(train_matrix)

    eig_pairs = [(np.abs(S[i]), VT[i]) for i in range(S.shape[0])]

    eig_pairs.sort(key=lambda arr: arr[0], reverse=True)
    # print(eig_pairs[:k])
    features = []
    for i in range(k):
        features.append(eig_pairs[i][1])

    # index_arr = np.argpartition(S, -k)[-k:]
    # features = VT[index_arr]
    # print(S[index_arr[:k]])
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
        train_label_reduced, test_label_reduced = svd(train_label, test_label, k)
        from oneNN import oneNN
        acc = oneNN(train_label_reduced, test_label_reduced)
        print("dataset reduced to {!s}, svd test acc is {!s}".format(k, acc))
