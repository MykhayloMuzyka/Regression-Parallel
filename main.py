import numpy as np
from multiprocessing import Manager
import concurrent.futures as futures


def ops(args):
    for item in args:
        if item[0] == 1:
            listToSum, power, res = item[1:]
            powerList = [i ** power for i in listToSum]
            res[power] = sum(powerList)
        else:
            X, Y, power, res = item[1:]
            powerX = [i ** power for i in X]
            powerList = [powerX[i] * Y[i] for i in range(len(X))]
            res[-(power + 1)] = sum(powerList)


def matrix_elements(args):
    listToSum, power, res = args
    powerList = [i ** power for i in listToSum]
    res[power] = sum(powerList)


def vector_elements(args):
    X, Y, power, res = args
    powerX = [i ** power for i in X]
    powerList = [powerX[i] * Y[i] for i in range(len(X))]
    res[-(power + 1)] = sum(powerList)


def regression(x: list, y: list, order: int):
    sumList = [0 for _ in range((order * 2 + 1) + (order + 1))]
    args = [(x, i, sumList) for i in range(order * 2 + 1)]
    for arg in args:
        matrix_elements(arg)
    args = [(x, y, i, sumList) for i in range(order + 1)]
    for arg in args:
        vector_elements(arg)
    a, b = [], []
    for i in range(order + 1):
        rowA = []
        for j in range(order + 1):
            rowA.append(sumList[i + j])
        a.append(rowA)
        b.append(sumList[-(i + 1)])
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return np.linalg.solve(a, b)


def regression_parallel(x: list, y: list, order: int, numOfProcesses: int):
    a, b = [], []
    sumList = Manager().list([0 for _ in range((order * 2 + 1) + (order + 1))])
    with futures.ProcessPoolExecutor(max_workers=numOfProcesses) as executor:
        args = [(1, x, i, sumList) for i in range(order * 2 + 1)] + [(2, x, y, i, sumList) for i in range(order + 1)]
        chunks = [int(i) for i in np.linspace(0, len(args), numOfProcesses + 1)]
        processes = []
        for i in range(numOfProcesses):
            processes.append(
                executor.submit(ops, args[chunks[i]:chunks[i + 1]])
            )
        while not all([i.done() for i in processes]):
            pass
    for i in range(order + 1):
        rowA = []
        for j in range(order + 1):
            rowA.append(sumList[i + j])
        a.append(rowA)
        b.append(sumList[-(i + 1)])
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return np.linalg.solve(a, b)


def f(x: float, coffs: list):
    res = 0
    for i in range(len(coffs)):
        res += coffs[i] * (x ** i)
    return res
