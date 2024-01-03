#!/usr/bin/python3
# Simplex method
# 作者: HongXin
# 2016.11.17

import numpy as np


def xlpsol(c, A, b):
    """
    解線性規劃問題，其格式如下：
    最小化  c^Tx
    限制條件: Ax <= b
               x >= 0
    (c^T 代表向量 c 的轉置)
    :return: x - 最佳解，opt - 最佳目標值
    """
    (B, T) = __init(c, A, b)
    (m, n) = T.shape
    opt = -T[0, 0]  # -T[0, 0] 正是最佳值!
    v_c = T[0, 1:]
    v_b = T[1:, 0]
    v_A = T[1:,1:]

    while True:
        if all(T[0, 1:] >= 0):  # c >= 0
            # 透過操作索引和值來獲取最佳解
            x = map(lambda t: T[B.index(t) + 1, 0] if t in B else 0,
                    range(0, n - 1))
            return x, opt
        else:
            # 選擇 v_c 中第一個小於 0 的元素
            e = next(x for x in v_c if x < 0)
            delta = map(lambda i: v_b[i]/v_A[i, e] , range(0, m-1))


def __init(c, A, b):
    """
    0   c   0
    b   A   I
    """
    # 轉換為向量和矩陣
    (c, A, b) = map(lambda t: np.array(t), [c, A, b])
    [m, n] = A.shape
    if m != b.size:
        print('b 的大小必須與 A 的行數相等！')
        exit(1)
    if n != c.size:
        print('c 的大小必須與 A 的列數相等！')
        exit(1)
    part_1 = np.vstack((0, b.reshape(b.size, 1)))
    part_2 = np.vstack((c, A))
    part_3 = np.vstack((np.zeros(m), np.identity(m)))
    return range(n, n + m), np.hstack((np.hstack((part_1, part_2)), part_3))


def __pivot():
    pass


if __name__ == '__main__':
    c = [-1, -14, -6]
    A = [[1, 1, 1], [1, 0, 0], [0, 0, 1], [0, 3, 1]]
    b = [4, 2, 3, 6]
    [x, opt] = xlpsol(c, A, b)
