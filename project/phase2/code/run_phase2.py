#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from python_log_indenter import IndentedLoggerAdapter
import statsmodels.api as sm


def readInData(filename):
    f = open(filename, 'rb')
    reader = csv.reader(f)
    data = np.array(list(reader)).astype('float')
    f.close()
    return data


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
    error = np.zeros(len(input))
    for (i, x) in enumerate(input):
        predict = beta[0] + beta[1] * x
        error[i] = target[i] - predict
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
        log.add().debug('Parameter for sensor %d: %s, %.4f' %
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


def windowInferHourStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data"""
    """Beta = m by 2"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    for j in range(0, n):
        if j == 0:
            Prediction[:, j] = InitMean
        else:
            for i in range(0, m):
                Prediction[i][j] = Beta[i][0] + \
                    Beta[i][1] * Prediction[i][j-1]

        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index][j] = Test[index][j]

        Error[:, j] = Test[:, j] - Prediction[:, j]
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
        if j == 0:
            Prediction[:, j] = InitMean
            MarginalVar[:, j] = InitVar
        else:
            for i in range(0, m):
                Prediction[i][j] = Beta[i, 0] + Beta[i, 1] * Prediction[i, j-1]
                if i in max_indices:
                    """previously observed sensor has zero variance"""
                    MarginalVar[i][j] = CondVar[i]
                else:
                    MarginalVar[i][j] = CondVar[i] + \
                        (Beta[i][1] ** 2) * MarginalVar[i][j-1]

        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index][j] = Test[index][j]
            MarginalVar[index][j] = 0

        Error[:, j] = Test[:, j] - Prediction[:, j]
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
    log.add().info('Hour stationary assumption')
    """learn model's parameters"""
    B, V, IM, IV = learnModelHourStationary(train_data)

    """inference and calculate error"""
    win_errors = np.zeros(len(budget_cnts))
    var_errors = np.zeros(len(budget_cnts))
    for i in range(0, len(budget_cnts)):
        win_errors[i] = windowInferHourStationary(B, IM,
                                                  test_data, budget_cnts[i])
        log.add().info('Avg Window error =\t%.4f with %d budget' %
                       (win_errors[i], budget_cnts[i]))
        log.sub()
        var_errors[i] = varianceInferHourStationary(B, V, IM, IV,
                                                    test_data, budget_cnts[i])
        log.add().info('Avg Variance error =\t%.4f with %d budget' %
                       (var_errors[i], budget_cnts[i]))
        log.sub()

    log.sub()
    return win_errors, var_errors


def learnModelDayStationary(train_data, n=48, m=50):
    """Beta = m * n * 2, [beta0, beta1] of sensor i"""
    """Variance = m * n, conditional variance X[j+1]|X[j] of sensor i"""
    """InitMean = m length list, initial mean of sensor i"""
    """InitVar = m length list, initial variance of sensor i"""
    Beta = np.zeros((m, n, 2))
    Variance = np.zeros((m, n))
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    for i in range(0, m):
        for j in range(0, n):
            if j < n - 1:
                X = [train_data[i][j], train_data[i][j + n],
                     train_data[i][j + 2*n]]
                y = [train_data[i][j + 1], train_data[i][j + n + 1],
                     train_data[i][j + 2*n + 1]]
            else:
                """learn parameter for day-to-day boundary"""
                X = [train_data[i][j], train_data[i][j + n]]
                y = [train_data[i][j + 1], train_data[i][j + n + 1]]

            constX = sm.add_constant(X)
            model = sm.OLS(y, constX)
            results = model.fit()
            Beta[i, j, :] = results.params
            Variance[i][j] = regressionError(Beta[i, j, :], X, y)
            """learn initial mean and variance"""
            if j == 0:
                InitMean[i] = np.mean(X)
                InitVar[i] = np.var(X)
                log.add().debug('Init for sensor %d: %.4f, %.4f'
                                % (i, InitMean[i], InitVar[i]))
                log.sub()

            log.add().debug('Parameter for sensor %d from %d to %d: %s, %.4f'
                            % (i, j, j + 1, str(Beta[i, j, :]), Variance[i][j]))
            log.sub()

    return Beta, Variance, InitMean, InitVar


def windowInferDayStationary(Beta, InitMean, Test, budget, n=96, m=50):
    """Test = m by n, test data"""
    """Beta = m by n by 2"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    day = n / 2
    for j in range(0, n):
        """start of days use init mean"""
        if j == 0:
            Prediction[:, j] = InitMean
        else:
            for i in range(0, m):
                t = (j-1) % day
                Prediction[i, j] = Beta[i, t, 0] +\
                    Beta[i, t, 1] * Prediction[i, j-1]

        """replace prediction with test data inside window"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        for k in range(window_start, window_end):
            index = k % m
            Prediction[index, j] = Test[index, j]

        Error[:, j] = Test[:, j] - Prediction[:, j]
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
    """Test = m * n, test data"""
    """Beta = m by (n-1) by 2"""
    """CondVar = m by (n-1)"""
    """InitMean and InitVar = m length list"""
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    day = n / 2
    max_indices = []
    for j in range(0, n):
        log.add().debug('%s are previously observed' % str(max_indices))
        log.sub()
        if j == 0:
            Prediction[:, j] = InitMean
            MarginalVar[:, j] = InitVar
        else:
            for i in range(0, m):
                t = (j-1) % day
                Prediction[i][j] = Beta[i, t, 0] +\
                Beta[i, t, 1] * Prediction[i, j-1]
                """previously observed sensor has zero marginal variance"""
                MarginalVar[i][j] = CondVar[i, t] + \
                        (Beta[i, t, 1] ** 2) * MarginalVar[i, j-1]

        max_indices = findLargestK(MarginalVar[:, j], budget, m)
        for index in max_indices:
            Prediction[index, j] = Test[index, j]
            MarginalVar[index, j] = 0

        Error[:, j] = Test[:, j] - Prediction[:, j]
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
    log.add().info('Day stationary assumption')
    """learn model's parameters"""
    B, V, IM, IV = learnModelDayStationary(train_data)
    """inference and calculate error"""
    win_errors = np.zeros(len(budget_cnts))
    var_errors = np.zeros(len(budget_cnts))
    for i in range(0, len(budget_cnts)):
        win_errors[i] = windowInferDayStationary(B, IM,
                                                 test_data, budget_cnts[i])
        log.add().info('Avg Window error =\t%.4f with %d budget' %
                       (win_errors[i], budget_cnts[i]))
        log.sub()
        var_errors[i] = varianceInferDayStationary(B, V, IM, IV,
                                                   test_data, budget_cnts[i])
        log.add().info('Avg Variance error =\t%.4f with %d budget' %
                       (var_errors[i], budget_cnts[i]))
        log.sub()

    log.sub()
    return win_errors, var_errors


def main(train_file, test_file):
    """ train_data = m * (3 day) shape matrix """
    train_data = readInData(train_file)

    """ test_data = m * (2 day) shape matrix """
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
    budget_cnts = [0, 5, 10, 20, 25]

    log.info('Processing temperature')
    topic = 'temperature'
    p2_h_win, p2_h_var, p2_d_win, p2_d_var = \
        main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    p1_win = [1.167, 1.049, 0.938, 0.692, 0.597]
    p1_var = [1.167, 0.967, 0.810, 0.560, 0.435]
    plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var,
                 p2_d_win, p2_d_var, y_max=3.5)
    log.info(str(p2_h_win))
    log.info(str(p2_h_var))
    log.info(str(p2_d_win))
    log.info(str(p2_d_var))

    log.info('Processing humidity')
    budget_cnts = [0, 5, 10, 20, 25]
    topic = 'humidity'
    p2_h_win, p2_h_var, p2_d_win, p2_d_var = \
        main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    p1_win = [3.470, 3.119, 2.782, 2.070, 1.757]
    p1_var = [3.470, 3.160, 2.847, 2.172, 1.822]
    plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var,
                 p2_d_win, p2_d_var)
    log.info(str(p2_h_win))
    log.info(str(p2_h_var))
    log.info(str(p2_d_win))
    log.info(str(p2_d_var))
