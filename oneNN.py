# -*- coding: utf-8 -*
import numpy as np


def oneNN(train, test):
    right_num = 0
    for arr in test:
        dist = np.sqrt(np.sum(np.square(arr[:-1] - train[0, :-1])))
        # dist = np.sqrt(np.linalg.norm(arr[:-1] - train[0, :-1]))
        label = train[0, -1]
        for i in range(1, train.shape[0]):
            temp_dist = np.sqrt(np.sum(np.square(arr[:-1] - train[i, :-1])))
            if temp_dist < dist:
                dist = temp_dist
                label = train[i, -1]
        # print("label: " + str(int(label)))
        # print("arr: " + str(int(arr[-1])))
        if label == arr[-1]:
            right_num += 1
    
    acc = right_num / (test.shape[0] * 1.0)
    return acc

