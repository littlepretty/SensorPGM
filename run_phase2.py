#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from python_log_indenter import IndentedLoggerAdapter
import statsmodels.api as sm


def regressionError(beta, input, target):
    error = np.zeros(len(input))
    i = 0
    for x in input:
        predict = beta[0] + beta[1] * x
        error[i] = target[i] - predict
        i += 1
    return np.var(error)


def learnModelAllStationary(train_data, n=48, m=50):
    """Beta = m * 2 shape matrix, Beta[i] = [beta0, bet1] of sensor i"""
    """Variance = m length list, standard deviation X_j+1|X_j"""
    Beta = np.zeros((m, 2))
    Variance = np.zeros(m)
    three_day = n * 3
    for i in range(0, m):
        X = train_data[i][:three_day - 1]
        y = train_data[i][1:]
        constX = sm.add_constant(X)
        model = sm.OLS(y, constX)
        results = model.fit()
        Beta[i, :] = results.params
        Variance[i] = regressionError(Beta[i, :], X, y)
        log.add().debug('Parameter for sensor %d: %s, %2f' %
                        (i, str(Beta[i, :]), Variance[i]))
        log.sub()

    """InitMean = m length list, initial mean"""
    """InitVar = m length list, initial variance"""
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        InitMean[i] = np.mean(train_data[i, :])
        InitVar[i] = np.var(train_data[i, :])

    return Beta, Variance, InitMean, InitVar


def learnModelDayStationary(train_data, n=48, m=50):
    """Beta = m * (n-1) * 2, [beta0, beta1] of sensor i"""
    """Variance = m * (n-1), conditional variance X_j+1|X_j of sensor i"""
    """InitMean = m length list, initial mean of sensor i"""
    """InitVar = m length list, initial variance of sensor i"""
    Beta = np.zeros((m, n-1, 2))
    Variance = np.zeros((m, n-1))
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        for j in range(0, n - 1):
            X = [train_data[i][j], train_data[i][j + n],
                 train_data[i][j + 2*n]]
            y = [train_data[i][j + 1], train_data[i][j + n + 1],
                 train_data[i][j + 2*n + 1]]
            constX = sm.add_constant(X)
            model = sm.OLS(y, constX)
            results = model.fit()
            Beta[i, j, :] = results.params
            Variance[i][j] = regressionError(Beta[i, j, :], X, y)
            if j == 0:
                InitMean[i] = np.mean(X)
                InitVar[i] = np.var(X)
            log.add().debug('Parameter for sensor %d from %d to %d: %s, %.2f' %
                            (i, j, j + 1, str(Beta[i, j, :]), Variance[i][j]))
            log.sub()
    return Beta, Variance, InitMean, InitVar


def windowInferAllStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m * n, test data"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    for j in range(0, n):
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
            else:
                Prediction[i][j] = Beta[i][0] + Beta[i][1] * Prediction[i-1][j]
        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index][j] = Test[index][j]

        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])
    avg_error = np.sum(Error) / (m * n)
    return avg_error


def varianceInferAllStationary(Beta, Variance,
                               InitMean, InitVariance,
                               Test, budget, n=96, m=50):
    """Test = m * n, test data"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    for j in range(0, n):
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
            else:
                Prediction[i][j] = Beta[i][j][0] +\
                    Beta[i][j][1] * Prediction[i-1][j]



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


def main(train_file, test_file):
    f = open(train_file, 'rb')
    reader = csv.reader(f)
    """train_data = m * (3 day) shape matrix"""
    train_data = np.array(list(reader)).astype('float')
    f.close()

    f = open(test_file, 'rb')
    reader = csv.reader(f)
    """test_data = m * (2 day) shape matrix"""
    test_data = np.array(list(reader)).astype('float')
    f.close()

    B, V, IM, IV = learnModelAllStationary(train_data)
    win_errors = [0] * len(budget_cnts)
    for i in range(0, len(budget_cnts)):
        win_errors[i] = windowInferAllStationary(B, IM,
                                                 test_data, budget_cnts[i])
        log.add().info('Avg error = %.2f with %d budget' %
                       (win_errors[i], budget_cnts[i]))
        log.sub()
    learnModelDayStationary(train_data)


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    log.info('Processing temperature')
    budget_cnts = [0, 5, 10, 20, 25]
    topic = 'temperature'
    main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
