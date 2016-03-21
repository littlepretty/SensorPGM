#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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


def windowInferenceError(day, means, b_cnt, n=96, m=50):
    error = []
    f = open('w%d.csv' % b_cnt, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    infer_data = np.zeros((m, n))
    for i in range(0, n):
        test_data = day[:, i]
        infer_data[:, i] = means[:, i % 48]
        window_start = int(i * b_cnt) % m
        window_size = b_cnt
        log.debug(str(range(window_start, window_start + window_size)))
        """replace inferred data with test data for these inside window"""
        for k in range(window_start, window_start + window_size):
            index = k % m
            infer_data[index, i] = test_data[index]
        """absolute error for time i"""
        error_i = np.subtract(test_data, infer_data[:, i])
        error_i = np.absolute(error_i)
        error.append(error_i)

    for i in range(0, m):
        row = [x for x in infer_data[i, :]]
        row.insert(0, i)
        writer.writerow(row)
    return error


def findLargestK(error, budget, m=50):
    max_indices = []
    indices = range(0, m)
    log.debug(str(error))
    for index in indices:
        if len(max_indices) == budget:
            break
        count = 0
        for j in range(0, m):
            if error[index] > error[j]:
                count += 1
        if count >= m - budget:
            max_indices.append(index)

    log.debug('read sensors %s' % str(max_indices))
    log.debug('#sensors = %d' % len(max_indices))
    return max_indices


def varianceInferenceError(day, means, stdevs, b_cnt, n=96, m=50):
    error = []
    f = open('v%d.csv' % b_cnt, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    infer_data = np.zeros((m, n))
    for i in range(0, n):
        test_data = day[:, i]
        infer_data[:, i] = means[:, i % 48]
        """find maximum variances' index"""
        variance = stdevs[:, i % 48]
        max_indices = findLargestK(variance, b_cnt, m)
        """replace most variant data with test data"""
        for index in max_indices:
            infer_data[index, i] = test_data[index]
        """absolute error for time i"""
        error_i = np.subtract(test_data, infer_data[:, i])
        error_i = np.absolute(error_i)
        error.append(error_i)

    for i in range(0, m):
        row = [x for x in infer_data[i, :]]
        row.insert(0, i)
        writer.writerow(row)
    return error


def inferenceTest(filename, means, stdevs, n=96, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')

    win_avg_errors = []
    var_avg_errors = []
    for cnt in budget_cnts:
        total_err = windowInferenceError(data, means, cnt)
        win_avg_err = np.sum(total_err) / (len(total_err) * len(total_err[0]))
        log.info('Window Inference for %.2f budget' % cnt)
        log.debug('error matrix \n' + str(total_err))
        log.add().info('avg error = ' + str(win_avg_err))
        log.sub()
        win_avg_errors.append(win_avg_err)

        total_err = varianceInferenceError(data, means, stdevs, cnt)
        var_avg_err = np.sum(total_err) / (len(total_err) * len(total_err[0]))
        log.info('Variance Inference for %.2f budget' % cnt)
        log.debug('error matrix \n' + str(total_err))
        log.add().info('avg error = ' + str(var_avg_err))
        log.sub()
        var_avg_errors.append(var_avg_err)

    return win_avg_errors, var_avg_errors


def plotAvgError(win, var):
    matplotlib.rc('font', size=18)
    index = np.arange(len(budget_cnts))
    bar_width = 0.27
    fig, ax = plt.subplots()
    rect1 = ax.bar(index, win, bar_width, color='b', hatch='/')
    rect2 = ax.bar(index + bar_width, var, bar_width, color='r', hatch='\\')
    ax.set_xlim([-0.5, 5])
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Budget Count')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(('0', '5', '10', '20', '25'))
    ax.legend((rect1[0], rect2[0]), ('Window', 'Variance'))
    plt.savefig('%s_err.eps' % topic, format='eps',
                bbox_inches='tight')
    # plt.show()


def main(train_file, test_file):
    means, stdevs = learnModel(train_file)
    win, var = inferenceTest(test_file, means, stdevs)
    plotAvgError(win, var)


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    title = ['sensors', 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
             11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
             16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
             21.5, 22.0, 22.5, 23.0, 23.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
             3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
             9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
             14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5,
             19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 0.0]
    budget_cnts = [20]
    budget_cnts = [0, 5, 10, 20, 25]
    log.info('Processing Temperature')
    log.add()
    topic = 'temperature'
    main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    log.sub()

    log.info('Processing Humidity')
    log.add()
    topic = 'humidity'
    main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    log.sub()
