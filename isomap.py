# -*- coding: utf-8 -*
import numpy as np


def mds(dist, red_dim):
    length = len(dist)

    dist_squa = dist ** 2

    dist_sum = np.sum(dist_squa, axis=1)
    dist_i = 1.0 / length * dist_sum

    dist_sum = np.sum(dist_squa, axis=0)
    dist_j = 1.0 / length * dist_sum

    dist_sum = np.sum(dist_squa)
    dist_ij = 1.0 / (length * length) * dist_sum

    B = np.ones([length, length], np.float32)
    for i in range(length):
        for j in range(length):
            B[i][j] = -0.5 * (dist_squa[i][j] - dist_i[i] - dist_j[j] + dist_ij)

    eig_values, eig_vectors = np.linalg.eig(B)
    index_arr = np.argpartition(eig_values, -red_dim)[-red_dim:]
    eig_values_diag = np.diag(eig_values[index_arr])
    res = np.matmul(eig_vectors[:, index_arr], np.sqrt(eig_values_diag))
    return res


def isomap_algorithm(data, k, red_dim):
    length = len(data)

    # 求距离矩阵
    dist_matrix = np.zeros([length, length], np.float32)
    for i in range(length):
        for j in range(length):
            dist_matrix[i][j] = np.linalg.norm(data[i] - data[j])

    # 求k近邻
    knn = np.ones([length, length], np.float32) * float('inf')
    for i in range(length):
        index_arr = np.argsort(dist_matrix[i])[:k]
        for j in index_arr:
            knn[i][j] = dist_matrix[i][j]

    # 最短路
    import scipy.sparse.csgraph as cg
    dist = cg.shortest_path(knn, directed=False)

    return mds(np.asarray(dist), red_dim)


def isomap(train_label, test_label, kpara, k):
    test_matrix = test_label[:, :-1]
    train_matrix = train_label[:, :-1]

    train_matrix_reduced = isomap_algorithm(train_matrix, kpara, k)
    test_matrix_reduced = isomap_algorithm(test_matrix, kpara, k)

    train_label_reduced = np.c_[train_matrix_reduced, train_label[:, -1]]
    test_label_reduced = np.c_[test_matrix_reduced, test_label[:, -1]]

    return train_label_reduced, test_label_reduced


if __name__ == '__main__':
    # train_label = np.loadtxt('./two_datasets/sonar-train.txt', delimiter=',')
    # test_label = np.loadtxt('./two_datasets/sonar-test.txt', delimiter=',')
    train_label = np.loadtxt('./two_datasets/splice-train.txt', delimiter=',')
    test_label = np.loadtxt('./two_datasets/splice-test.txt', delimiter=',')

    kpara = 8
    k_arr = [10, 20, 30]
    for k in k_arr:
        train_label_reduced, test_label_reduced = isomap(train_label, test_label, kpara, k)

        from oneNN import oneNN
        acc_pca = oneNN(train_label_reduced, test_label_reduced)
        print("dataset reduced to {!s}, isomap test acc is {!s}".format(k, acc_pca))
