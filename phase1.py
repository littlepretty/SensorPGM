#!/usr/bin/env python

import csv
import logging
import numpy


def learnModel(filename, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)

    day1 = []
    day2 = []
    day3 = []
    for row in reader:
        day1.append([float(x) for x in row[:n]])
        day2.append([float(x) for x in row[n:n*2]])
        day3.append([float(x) for x in row[n*2:n*3]])

    """learn parameters for m*n random variables"""
    means = [[0 for _ in range(0, n)] for _ in range(0, m)]
    devs = [[0 for _ in range(0, n)] for _ in range(0, m)]
    for i in range(0, m):
        for j in range(0, n):
            row = [day1[i][j], day2[i][j], day3[i][j]]
            means[i][j] = numpy.mean(row)
            devs[i][j] = pow(numpy.std(row), 2) / (len(row) - 1)

    logging.debug(means[:1])
    logging.debug(devs[:1])


def inferenceTest(filename, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = numpy.array(list(reader)).astype('float')
    logging.debug(data)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    learnModel('intelTemperatureTrain.csv')
    inferenceTest('intelTemperatureTest.csv')
