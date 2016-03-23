#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from python_log_indenter import IndentedLoggerAdapter
import statsmodels.api as sm


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


def plotAvgError(p1_win, p1_var,
                 p2_all_win, p2_all_var,
                 p2_day_win, p2_day_var, y_max=8):
    matplotlib.rc('font', size=18)
    width = 2
    index = np.arange(0, 7 * width * len(budget_cnts), width * 7)
    fig, ax = plt.subplots()
    rect1 = ax.bar(index, p1_win, width, color='b', hatch='/')
    rect2 = ax.bar(index + width, p1_var, width, color='r', hatch='\\')
    rect3 = ax.bar(index + width*2, p2_all_win, width, color='g', hatch='-')
    rect4 = ax.bar(index + width*3, p2_all_var, width, color='c', hatch='/')
    rect5 = ax.bar(index + width*4, p2_day_win, width, color='m', hatch='\\')
    rect6 = ax.bar(index + width*5, p2_day_var, width, color='y', hatch='-')
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
    error = np.zeros(len(input))
    i = 0
    for x in input:
        predict = beta[0] + beta[1] * x
        error[i] = target[i] - predict
        i += 1
    return np.var(error)


def learnModelHourStationary(train_data, n=48, m=50):
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


def windowInferHourStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data"""
    """Beta = m by 2"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    for j in range(0, n):
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
            else:
                Prediction[i][j] = Beta[i][0] + Beta[i][1] * Prediction[i][j-1]
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


def windowInferDayStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data"""
    """Beta = m by (n-1) by 2"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    day = n / 2
    for j in range(0, n):
        for i in range(0, m):
            """start of days use init mean"""
            if j % day == 0:
                Prediction[i, j] = InitMean[i]
            else:
                t = (j-1) % day
                Prediction[i, j] = Beta[i, t, 0] +\
                    Beta[i, t, 1] * Prediction[i, t]
        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index, j] = Test[index, j]

        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])
    avg_error = np.sum(Error) / (m * n)
    return avg_error


def varianceInferHourStationary(Beta, Var, InitMean, InitVar,
                                Test, budget, n=96, m=50):
    """Test = m * n, test data"""
    """Beta = m by 2"""
    """Var = m length list"""
    """InitMean and InitVar = m length list"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    for j in range(0, n):
        for i in range(0, m):
            if j == 0:
                Prediction[i][j] = InitMean[i]
                MarginalVar[i][j] = InitVar[i]
            else:
                Prediction[i][j] = Beta[i, 0] + Beta[i, 1] * Prediction[i, j-1]
                MarginalVar[i][j] = Var[i] + \
                    (Beta[i][1] ** 2) * MarginalVar[i][j-1]
        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index][j] = Test[index][j]
        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])
    avg_error = np.sum(Error) / (m * n)
    return avg_error


def varianceInferDayStationary(Beta, Var, InitMean, InitVar,
                               Test, budget, n=96, m=50):
    """Test = m * n, test data"""
    """Beta = m by (n-1) by 2"""
    """Var = m by (n-1)"""
    """InitMean and InitVar = m length list"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    day = n / 2
    for j in range(0, n):
        for i in range(0, m):
            if j % day == 0:
                Prediction[i][j] = InitMean[i]
                MarginalVar[i][j] = InitVar[i]
            else:
                t = (j-1) % day
                Prediction[i][j] = Beta[i, t, 0] +\
                    Beta[i, t, 1] * Prediction[i, j-1]
                MarginalVar[i][j] = Var[i, t] + \
                    (Beta[i, t, 1] ** 2) * MarginalVar[i, j-1]
        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index][j] = Test[index][j]
        Error[:, j] = np.subtract(Test[:, j], Prediction[:, j])
        Error[:, j] = np.absolute(Error[:, j])
    avg_error = np.sum(Error) / (m * n)
    return avg_error


def hourStationary(train_data, test_data):
    log.add().info('Hour stationary assumption')
    """learn model's parameters"""
    B, V, IM, IV = learnModelHourStationary(train_data)

    """inference and calculate error"""
    win_errors = [0] * len(budget_cnts)
    var_errors = [0] * len(budget_cnts)
    for i in range(0, len(budget_cnts)):
        win_errors[i] = windowInferHourStationary(B, IM,
                                                  test_data, budget_cnts[i])
        log.add().info('Avg Window error = %.2f with %d budget' %
                       (win_errors[i], budget_cnts[i]))
        log.sub()

        var_errors[i] = varianceInferHourStationary(B, V, IM, IV,
                                                    test_data, budget_cnts[i])
        log.add().info('Avg Variance error = %.2f with %d budget' %
                       (var_errors[i], budget_cnts[i]))
        log.sub()
    log.sub()
    return win_errors, var_errors


def dayStationary(train_data, test_data):
    log.add().info('Day stationary assumption')
    """learn model's parameters"""
    B, V, IM, IV = learnModelDayStationary(train_data)
    """inference and calculate error"""
    win_errors = [0] * len(budget_cnts)
    var_errors = [0] * len(budget_cnts)
    for i in range(0, len(budget_cnts)):
        win_errors[i] = windowInferDayStationary(B, IM,
                                                 test_data, budget_cnts[i])
        log.add().info('Avg Window error = %.2f with %d budget' %
                       (win_errors[i], budget_cnts[i]))
        log.sub()

        var_errors[i] = varianceInferDayStationary(B, V, IM, IV,
                                                   test_data, budget_cnts[i])
        log.add().info('Avg Variance error = %.2f with %d budget' %
                       (var_errors[i], budget_cnts[i]))
        log.sub()

    log.sub()
    return win_errors, var_errors


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
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    log.info('Processing temperature')
    budget_cnts = [0, 5, 10, 20, 25]
    topic = 'temperature'
    p2_h_win_err, p2_h_var_err, p2_d_win_err, p2_d_var_err = \
        main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    t_p1_win_err = [1.167, 1.049, 0.938, 0.692, 0.597]
    t_p1_var_err = [1.167, 0.967, 0.810, 0.560, 0.435]
    plotAvgError(t_p1_win_err, t_p1_var_err,
                 p2_h_win_err, p2_h_var_err,
                 p2_d_win_err, p2_d_var_err, y_max=4)

    log.info('Processing humidity')
    budget_cnts = [0, 5, 10, 20, 25]
    topic = 'humidity'
    p2_h_win_err, p2_h_var_err, p2_d_win_err, p2_d_var_err = \
        main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    h_p1_win_err = [3.470, 3.119, 2.782, 2.070, 1.757]
    h_p1_var_err = [3.470, 3.160, 2.847, 2.172, 1.822]
    plotAvgError(h_p1_win_err, h_p1_var_err,
                 p2_h_win_err, p2_h_var_err,
                 p2_d_win_err, p2_d_var_err)
