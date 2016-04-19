#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from python_log_indenter import IndentedLoggerAdapter
import statsmodels.api as sm


def readInData(filename):
    """read in numpy data from csv file"""
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')
    f.close()
    return data


def plotAvgError(p1_win, p1_var,
                 p2_all_win, p2_all_var,
                 p2_day_win, p2_day_var,
                 p3_all_win, p3_all_var,
                 p3_day_win, p3_day_var,
                 y_max=8):
    matplotlib.rc('font', size=18)
    width = 2
    fig, ax = plt.subplots()
    data_list = [p1_win, p1_var,
                 p2_all_win, p2_all_var,
                 p2_day_win, p2_day_var,
                 p3_all_win, p3_all_var,
                 p3_day_win, p3_day_var]
    labels = ('Phase 1 Window', 'Phase 1 Variance',
              'Phase 2 H-Window', 'Phase 2 H-Variance',
              'Phase 2 D-Window', 'Phase 2 D-Variance',
              'Phase 3 H-Window', 'Phase 3 H-Variance',
              'Phase 3 D-Window', 'Phase 3 D-Variance')
    hatches = ['/', '\\', '//', 'x', '+']
    colors = ['b', 'r', 'g', 'c', 'm', 'y']
    rects = []
    index_cnt = len(data_list) + 1
    index = np.arange(0, index_cnt * width * len(budget_cnts),
                      width * index_cnt)
    for (i, data) in enumerate(data_list):
        rect = ax.bar(index + i * width, data, width,
                      color=colors[i % len(colors)],
                      hatch=hatches[i % len(hatches)])
        rects.append(rect)

    ax.set_xlim([-3, index_cnt * width * (len(budget_cnts) + 0.1)])
    ax.set_ylim([0, y_max])
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Budget Count')
    ax.set_xticks(index + width * ((index_cnt) - 1) / 2)
    ax.set_xticklabels(('0', '5', '10', '20', '25'))
    ax.legend(rects, labels, ncol=2, fontsize=15)
    plt.grid()
    plt.savefig('%s_err.eps' % topic, format='eps',
                bbox_inches='tight')


def findLargestK(error, k, m=50):
    """return the indices of largest k valuse in list/error"""
    max_indices = []
    indices = range(0, m)
    log.debug(str(error))
    for index in indices:
        if len(max_indices) == k:
            break
        count = 0
        for j in range(0, m):
            if error[index] > error[j]:
                count += 1
        if count >= m - k:
            max_indices.append(index)

    log.debug('read sensors %s' % str(max_indices))
    log.debug('#sensors = %d' % len(max_indices))
    return max_indices


def regressionError(beta, input, target):
    """beta = [m+1] vector
    input =  m + 1
    """
    Predict = np.dot(beta, np.transpose(input))
    Error = np.subtract(Predict, target)
    return np.var(Error)


def learnModelHourStationary(train_data, n=48, m=50):
    """Beta = m * (m+1) matrix,
    Beta[i] = [beta_0, beta_1,...,beta_m] for sensor i]
    Variance = [m] list, variance of X[i,j+1]|X[1,j], X[2,j],...,X[m,j]"""
    Beta = np.zeros((m, m+1))
    Variance = np.zeros(m)
    three_day = n * 3
    for i in range(0, m):
        X = [train_data[:, j] for j in range(0, three_day-1)]
        y = train_data[i][1:]
        constX = sm.add_constant(X)
        model = sm.OLS(y, constX)
        results = model.fit()
        Beta[i, :] = results.params
        Variance[i] = regressionError(Beta[i, :], constX, y)
        log.add().debug('Parameter for sensor %d: %s, %.2f' %
                        (i, str(Beta[i, :]), Variance[i]))
        log.sub()

    """InitMean = m length list, initial mean"""
    """InitVar = m length list, initial variance"""
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        InitMean[i] = np.mean(train_data[i, :])
        InitVar[i] = np.var(train_data[i, :])

    log.add().debug('Init mean = %s' % str(InitMean))
    log.debug('Init variance = %s' % str(InitVar))
    log.sub()
    return Beta, Variance, InitMean, InitVar


def windowInferHourStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data"""
    """Beta = m by (m+1)"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    for j in range(0, n):
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
            else:
                X = np.insert(Prediction[:, j-1], 0, 1)
                Prediction[i, j] = np.dot(Beta[i, :], X)

        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index][j] = Test[index][j]

        log.add().debug('Window preciton at %d = %s'
                        % (j, Prediction[:, j]))
        log.sub()

        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])
    avg_error = np.sum(Error) / (m * n)

    f = open('h-w%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error


def varianceInferHourStationary(Beta, CondVar, InitMean, InitVar,
                                Test, budget, n=96, m=50):
    """Test = m * n, test data"""
    """Beta = m by 2"""
    """CondVar = m length list"""
    """InitMean and InitVar = m length list"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    max_indices = []
    for j in range(0, n):
        log.add().debug('%s are previously observed' % str(max_indices))
        log.sub()
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
                MarginalVar[i][j] = InitVar[i]
            else:
                X = np.insert(Prediction[:, j-1], 0, 1)
                Prediction[i][j] = np.dot(Beta[i, :], X)
                if i in max_indices:
                    """previously observed sensor has zero variance"""
                    MarginalVar[i][j] = CondVar[i]
                else:
                    BetaSquared = np.square(Beta[i, 1:])
                    MarginalVar[i][j] = CondVar[i] + \
                        np.dot(BetaSquared, MarginalVar[:, j-1])

        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index][j] = Test[index][j]
            MarginalVar[index][j] = 0

        log.add().debug('Window preciton at %d = %s'
                        % (j, Prediction[:, j]))
        log.sub()
        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])

    avg_error = np.sum(Error) / (m * n)

    f = open('h-v%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error


def hourStationary(train_data, test_data):
    win_errs = [0] * len(budget_cnts)
    var_errs = [0] * len(budget_cnts)
    log.add().info('Hour stationary assumption')
    B, V, IM, IV = learnModelHourStationary(train_data)
    for i in range(0, len(budget_cnts)):
        win_errs[i] = windowInferHourStationary(B, IM,
                                                test_data, budget_cnts[i])
        log.add().info('Avg Window error = %.2f with %d budget' %
                       (win_errs[i], budget_cnts[i]))
        log.sub()
        var_errs[i] = varianceInferHourStationary(B, V, IM, IV,
                                                  test_data, budget_cnts[i])
        log.add().info('Avg Variance error = %.2f with %d budget' %
                       (var_errs[i], budget_cnts[i]))
        log.sub()
    log.sub()
    return win_errs, var_errs


def learnModelDayStationary(train_data, n=48, m=50):
    Beta = np.zeros((m, n, m+1))
    Variance = np.zeros((m, n))
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        for j in range(0, n):
            if j < n - 1:
                """only 3 rows of regression data"""
                X = [train_data[:, j], train_data[:, j + n],
                     train_data[:, j + 2 * n]]
                y = [train_data[i, j + 1], train_data[i, j + n + 1],
                     train_data[i, j + 2 * n + 1]]
            else:
                X = [train_data[:, j], train_data[:, j + n]]
                y = [train_data[i, j + 1], train_data[i, j + n + 1]]

            constX = sm.add_constant(X)
            model = sm.OLS(y, constX)
            results = model.fit()
            Beta[i, j, :] = results.params
            Variance[i][j] = regressionError(Beta[i, j, :], constX, y)
            if j == 0:
                InitMean[i] = np.mean(X)
                InitVar[i] = np.var(X)

            log.add().debug('Parameter for sensor %d from %d to %d: %s, %.2f' %
                            (i, j, j + 1, str(Beta[i, j, :]), Variance[i][j]))
            log.sub()

    log.add().debug('Init mean = %s' % str(InitMean))
    log.debug('Init variance = %s' % str(InitVar))
    log.sub()
    return Beta, Variance, InitMean, InitVar


def windowInferDayStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data
    Beta = m by n/2 by 2 matrix
    InitMean = [m] list
    """
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    day = n / 2
    for j in range(0, n):
        for i in range(0, m):
            """start of days use init mean"""
            if j == 0:
                Prediction[i, j] = InitMean[i]
            else:
                t = (j-1) % day
                X = np.insert(Prediction[:, t], 0, 1)
                Prediction[i, j] = np.dot(Beta[i, t, :], X)

        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index, j] = Test[index, j]

        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])

    avg_error = np.sum(Error) / (m * n)

    f = open('d-w%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error


def varianceInferDayStationary(Beta, CondVar, InitMean, InitVar,
                               Test, budget, n=96, m=50):
    """Test = m * n, test data matrix
    Beta = m by n/2 by (m+1) matrix
    CondVar = m by n/2 matrix
    InitMean and InitVar = m length list"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    day = n / 2
    max_indices = []
    for j in range(0, n):
        log.add().debug('%s are previously observed' % str(max_indices))
        log.sub()
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
                MarginalVar[i][j] = InitVar[i]
            else:
                t = (j-1) % day
                X = np.insert(Prediction[:, j-1], 0, 1)
                Prediction[i][j] = np.dot(Beta[i, t, :], X)
                if i in max_indices:
                    """previously observed sensor has zero variance"""
                    MarginalVar[i][j] = CondVar[i, t]
                else:
                    BetaSquared = np.square(Beta[i, t, 1:])
                    MarginalVar[i][j] = CondVar[i, t] + \
                        np.dot(BetaSquared, MarginalVar[:, j-1])

        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index][j] = Test[index][j]
            MarginalVar[index][j] = 0
        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])

    avg_error = np.sum(Error) / (m * n)

    f = open('d-v%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error


def dayStationary(train_data, test_data):
    win_errs = [0] * len(budget_cnts)
    var_errs = [0] * len(budget_cnts)
    log.add().info('Day stationary assumption')
    B, V, IM, IV = learnModelDayStationary(train_data)
    for i in range(0, len(budget_cnts)):
        win_errs[i] = windowInferDayStationary(B, IM,
                                               test_data, budget_cnts[i])
        log.add().info('Avg Window error = %.2f with %d budget' %
                       (win_errs[i], budget_cnts[i]))
        log.sub()
        var_errs[i] = varianceInferDayStationary(B, V, IM, IV,
                                                 test_data, budget_cnts[i])
        log.add().info('Avg Variance error = %.2f with %d budget' %
                       (var_errs[i], budget_cnts[i]))
        log.sub()
    log.sub()
    return win_errs, var_errs


def main(train_file, test_file):
    """train_data = m * (3 day) shape matrix"""
    train_data = readInData(train_file)
    """test_data = m * (2 day) shape matrix"""
    test_data = readInData(test_file)

    h_win_err, h_var_err = hourStationary(train_data, test_data)
    d_win_err, d_var_err = dayStationary(train_data, test_data)

    return h_win_err, h_var_err, d_win_err, d_var_err


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    np.set_printoptions(precision=3)
    title = ['sensors', 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
             5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0,
             11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
             16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0,
             21.5, 22.0, 22.5, 23.0, 23.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
             3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5,
             9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5,
             14.0, 14.5, 15.0, 15.5, 16.0, 16.5, 17.0, 17.5, 18.0, 18.5,
             19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0, 22.5, 23.0, 23.5, 0.0]
    # budget_cnts = [0, 5]
    budget_cnts = [0, 5, 10, 20, 25]
    log.info('Processing temperature')
    topic = 'temperature'
    p3_h_win, p3_h_var, p3_d_win, p3_d_var = \
        main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    p1_win = [1.167, 1.049, 0.938, 0.692, 0.597]
    p1_var = [1.167, 0.967, 0.810, 0.560, 0.435]
    p2_h_win = [2.366, 1.561, 0.905, 0.412, 0.274]
    p2_h_var = [2.366, 1.519, 0.968, 0.454, 0.325]
    p2_d_win = [1.125, 0.94, 0.556, 0.279, 0.214]
    p2_d_var = [1.125, 0.774, 0.567, 0.295, 0.226]
    """plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var, p2_d_win, p2_d_var,
                 p3_h_win, p3_h_var, p3_d_win, p3_d_var)

    log.info('Processing humidity')
    topic = 'humidity'
    p3_h_win, p3_h_var, p3_d_win, p3_h_var = \
        main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    p1_win = [3.470, 3.119, 2.782, 2.070, 1.757]
    p1_var = [3.470, 3.160, 2.847, 2.172, 1.822]
    p2_h_win = [5.365, 2.701, 1.524, 0.689, 0.452]
    p2_h_var = [5.365, 2.964, 1.752, 0.786, 0.573]
    p2_d_win = [3.872, 2.327, 1.282, 0.718, 0.515]
    p2_d_var = [3.872, 2.123, 1.476, 0.744, 0.572]
    plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var, p2_d_win, p2_d_var,
                 p3_h_win, p3_h_var, p3_d_win, p3_d_var)"""
