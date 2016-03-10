#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
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
    f = open('w%d.csv' % int(percentage * 100), 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    infer_data = np.zeros((m, n))
    for i in range(0, n):
        test_data = day[:, i]
        # predict by generate random number? or just use mean
        infer_data[:, i] = means[:, i]

        window_start = int(i * m * percentage) % m
        window_size = int(m * percentage)
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


def varianceInferenceError(day, means, percentage, n=48, m=50):
    error = []
    f = open('w%d.csv' % int(percentage * 100), 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    infer_data = np.zeros((m, n))
    for i in range(0, n):
        test_data = day[:, i]
        infer_data[:, i] = means[:, i]
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


def inferenceTest(filename, means, n=48, m=50):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')
    day1 = data[:, 0:n]
    day2 = data[:, n:n*2]

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
        var_avg_errors.append(var_avg_err)

    return win_avg_errors, var_avg_errors


def plotAvgError(win, var):
    index = np.arange(len(percentage))
    bar_width = 0.27
    fig, ax = plt.subplots()
    rect1 = ax.bar(index, win, bar_width, color='b', hatch='/')
    rect2 = ax.bar(index + bar_width, var, bar_width, color='r', hatch='\\')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Budget Percentage')
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(('0%', '%5', '10%', '25%', '50%'))
    ax.legend((rect1[0], rect2[0]), ('Window', 'Variance'))
    plt.show()


def main(train_file, test_file):
    means, stdevs = learnModel(train_file)
    win, var = inferenceTest(test_file, means)
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
    percentage = [0, 0.05, 0.1, 0.25, 0.5]
    log.info('Processing Temperature')
    log.add()
    main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    log.sub()

    log.info('Processing Humidity')
    log.add()
    main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    log.sub()
