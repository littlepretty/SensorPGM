#!/usr/bin/env python

import csv
import logging as lg
import numpy as np


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
    # means = [[0 for _ in range(0, n)] for _ in range(0, m)]
    means = np.zeros((m, n))
    stdevs = np.zeros((m, n))
    # stdevs = [[0 for _ in range(0, n)] for _ in range(0, m)]
    for i in range(0, m):
        for j in range(0, n):
            row = [day1[i][j], day2[i][j], day3[i][j]]
            means[i][j] = np.mean(row)
            stdevs[i][j] = np.std(row) / np.sqrt(len(row) - 1)

    lg.debug(means[:1])
    lg.debug(stdevs[:1])
    return means, stdevs


def windowInferenceError(day, percentage, n=48, m=50):
    error = []
    for i in range(0, n):
        test_data = day[:, i]
        # predict by generate random number? or just use mean
        infer_data = means[:, i]

        window_start = int(i * m * percentage) % m
        window_size = int(m * percentage)

        """replace inferred data with test data for these inside window"""
        for k in range(window_start, window_start + window_size):
            index = k % m
            infer_data[index] = test_data[index]

        """absolute error for time i"""
        error_i = np.subtract(test_data, infer_data)
        error_i = np.absolute(error_i)
        error.append(error_i)
    return error


def varianceInferenceError(day, percentage, n=48, m=50):
    error = []
    for i in range(0, n):
        test_data = day[:, i]
        infer_data = means[:, i]
        if i != 0:
            """find maximum variances' index"""
            last_error = error[-1]
            lg.debug('last error \n' + str(last_error))
            num_var = int(n * percentage)
            max_indices = []
            for _ in range(0, num_var):
                max_index = 0
                for j in range(0, m):
                    if last_error[j] > last_error[max_index] \
                            and j not in max_indices:
                        max_index = j
                max_indices.append(max_index)
            """replace most variant data with test data"""
            for index in max_indices:
                infer_data[index] = test_data[index]

        """absolute error for time i"""
        error_i = np.subtract(test_data, infer_data)
        error_i = np.absolute(error_i)
        error.append(error_i)
    return error


def inferenceTest(filename, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')
    day1 = data[:, 0:n]
    day2 = data[:, n:n*2]
    percentage = 0.3

    day1_error = windowInferenceError(day1, percentage)
    day2_error = windowInferenceError(day2, percentage)
    total_error = np.add(day1_error, day2_error)
    avg_error = np.sum(total_error) / total_error.size
    lg.info('Window Inference')
    lg.debug('error matrix \n' + str(total_error))
    lg.info('avg error = ' + str(avg_error))

    day1_error = varianceInferenceError(day1, percentage)
    day2_error = varianceInferenceError(day2, percentage)
    total_error = np.add(day1_error, day2_error)
    avg_error = np.sum(total_error) / total_error.size
    lg.info('Variance Inference')
    lg.debug('error matrix \n' + str(total_error))
    lg.info('avg error = ' + str(avg_error))

    return avg_error


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    means, stdevs = learnModel('intelTemperatureTrain.csv')
    inferenceTest('intelTemperatureTest.csv')
