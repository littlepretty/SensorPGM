#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
from python_log_indenter import IndentedLoggerAdapter


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
    means = np.zeros((m, n))
    stdevs = np.zeros((m, n))
    for i in range(0, m):
        for j in range(0, n):
            row = [day1[i][j], day2[i][j], day3[i][j]]
            means[i][j] = np.mean(row)
            stdevs[i][j] = np.std(row) / np.sqrt(len(row) - 1)
    log.debug(str(means[:1]))
    log.debug(str(stdevs[:1]))
    return means, stdevs


def windowInferenceError(day, means, percentage, n=48, m=50):
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


def varianceInferenceError(day, means, percentage, n=48, m=50):
    error = []
    for i in range(0, n):
        test_data = day[:, i]
        infer_data = means[:, i]
        if i != 0:
            """find maximum variances' index"""
            last_error = error[-1]
            log.debug('last error \n' + str(last_error))
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


def inferenceTest(filename, means, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')
    day1 = data[:, 0:n]
    day2 = data[:, n:n*2]
    percentage = [0, 0.05, 0.1, 0.25, 0.5]
    win_avg_errors = []
    var_avg_errors = []
    for p in percentage:
        day1_err = windowInferenceError(day1, means, p)
        day2_err = windowInferenceError(day2, means, p)
        total_err = np.add(day1_err, day2_err)
        win_avg_err = np.sum(total_err) / total_err.size
        log.info('Window Inference for %.2f budget' % p)
        log.debug('error matrix \n' + str(total_err))
        log.add().info('avg error = ' + str(win_avg_err))
        log.sub()
        win_avg_errors.append(win_avg_err)

        day1_err = varianceInferenceError(day1, means, p)
        day2_err = varianceInferenceError(day2, means, p)
        total_err = np.add(day1_err, day2_err)
        var_avg_err = np.sum(total_err) / total_err.size
        log.info('Variance Inference for %.2f budget' % p)
        log.debug('error matrix \n' + str(total_err))
        log.add().info('avg error = ' + str(var_avg_err))
        log.sub()
        var_avg_errors.append(win_avg_err)

    return [win_avg_errors, var_avg_errors]


def main(train_file, test_file):
    means, stdevs = learnModel(train_file)
    (win_avg_err, var_avg_err) = inferenceTest(test_file, means)


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    log.info('Processing Temperature')
    log.add()
    main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    log.sub()

    log.info('Processing Humidity')
    log.add()
    main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    log.sub()
