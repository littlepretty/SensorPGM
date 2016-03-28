#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from python_log_indenter import IndentedLoggerAdapter
import statsmodels.api as sm


def plotAvgError(p1_win, p1_var,
                 p2_all_win, p2_all_var,
                 p2_day_win, p2_day_var, y_max=8):
    matplotlib.rc('font', size=18)
    width = 2
    index = np.arange(0, 7 * width * len(budget_cnts), width * 7)
    fig, ax = plt.subplots()
    rect1 = ax.bar(index, p1_win, width, color='b', hatch='/')
    rect2 = ax.bar(index + width, p1_var, width, color='r', hatch='\\')
    rect3 = ax.bar(index + width*2, p2_all_win, width, color='g', hatch='//')
    rect4 = ax.bar(index + width*3, p2_all_var, width, color='c', hatch='\\')
    rect5 = ax.bar(index + width*4, p2_day_win, width, color='m', hatch='x')
    rect6 = ax.bar(index + width*5, p2_day_var, width, color='y', hatch='//')
    ax.set_xlim([-3, 7 * width * (len(budget_cnts) + 0.1)])
    ax.set_ylim([0, y_max])
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Budget Count')
    ax.set_xticks(index + width * 2.5)
    ax.set_xticklabels(('0', '5', '10', '20', '25'))
    ax.legend((rect1[0], rect2[0], rect3[0], rect4[0], rect5[0], rect6[0]),
              ('Phase 1 Window', 'Phase 1 Variance',
               'Phase 2 H-Window', 'Phase 2 H-Variance',
               'Phase 2 D-Window', 'Phase 2 D-Variance'),
              ncol=2, fontsize=15)
    plt.grid()
    plt.savefig('%s_err.eps' % topic, format='eps',
                bbox_inches='tight')


def regressionError(beta, input, target):
    Predict = np.dot(beta, np.transpose(input))
    Error = np.subtract(Predict, target)
    return np.var(Error)


def learnModelHourStationary(train_data, n=48, m=50):
    """Beta = m * (m+1) matrix,
    Beta[i] = [beta0, beta1,...,beta_m] for sensor i]"""
    """Variance = [m] list, variance of X_i,j+1|X_1,j, X_2,j,...,X_M,j"""
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


def learnModelDayStationary(train_data, n=48, m=50):
    Beta = np.zeros((m, n, m+1))
    Variance = np.zeros((m, n))
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        for j in range(0, n):
            if j < n - 1:
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


def hourStationary(train_data, test_data):
    win_errs = [0] * len(budget_cnts)
    var_errs = [0] * len(budget_cnts)
    B, V, IM, IV = learnModelHourStationary(train_data)

    return win_errs, var_errs


def dayStationary(train_data, test_data):
    win_errs = [0] * len(budget_cnts)
    var_errs = [0] * len(budget_cnts)
    B, V, IM, IV = learnModelDayStationary(train_data)

    return win_errs, var_errs


def main(train_file, test_file):
    f = open(train_file, 'rb')
    reader = csv.reader(f)
    # train_data = m * (3 day) shape matrix
    train_data = np.array(list(reader)).astype('float')
    f.close()

    f = open(test_file, 'rb')
    reader = csv.reader(f)
    # test_data = m * (2 day) shape matrix
    test_data = np.array(list(reader)).astype('float')
    f.close()

    h_win_err, h_var_err = hourStationary(train_data, test_data)
    d_win_err, d_var_err = dayStationary(train_data, test_data)

    return h_win_err, h_var_err, d_win_err, d_var_err


if __name__ == '__main__':
    lg.basicConfig(level=lg.DEBUG)
    # lg.basicConfig(level=lg.INFO)
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
    # budget_cnts = [5]
    budget_cnts = [0, 5, 10, 20, 25]
    log.info('Processing temperature')
    topic = 'temperature'
    main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
