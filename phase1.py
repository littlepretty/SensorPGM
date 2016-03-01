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
    stdevs = [[0 for _ in range(0, n)] for _ in range(0, m)]
    for i in range(0, m):
        for j in range(0, n):
            row = [day1[i][j], day2[i][j], day3[i][j]]
            means[i][j] = numpy.mean(row)
            stdevs[i][j] = numpy.std(row) / numpy.sqrt(len(row) - 1)

    logging.debug(means[:1])
    logging.debug(stdevs[:1])
    return means, stdevs


def inferenceTest(filename, means, stdevs, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = numpy.array(list(reader)).astype('float')
    day1 = data[:, 0:n]
    day2 = data[:, n:n*2]

    percentage = 0.3
    for i in range(0, n):
        test_data = day1[:, i]
        infer_data = [0 for _ in range(0, m)]

        for j in range(0, m):
            infer_data[j] = numpy.random.normal(means[j][i], stdevs[j][i])

        window_start = int(i * m * percentage) % m
        window_size = int(m * percentage)

        for k in range(window_start, window_start + window_size):
            index = k % m
            infer_data[index] = test_data[index]

        error = numpy.subtract(test_data, infer_data)
        logging.debug('error \n'+ str(error))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    means, stdevs = learnModel('intelTemperatureTrain.csv')
    inferenceTest('intelTemperatureTest.csv', means, stdevs)
