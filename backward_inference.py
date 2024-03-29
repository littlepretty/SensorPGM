#!/usr/bin/env python

import csv
import logging as lg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from python_log_indenter import IndentedLoggerAdapter


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
                 y_max=8):
    matplotlib.rc('font', size=18)
    width = 2
    fig, ax = plt.subplots()
    data_list = [p1_win, p1_var,
                 p2_all_win, p2_all_var,
                 p2_day_win, p2_day_var,
                 p3_all_win, p3_all_var]
    labels = ('Phase 1 Window', 'Phase 1 Variance',
              'Phase 2 H-Window', 'Phase 2 H-Variance',
              'Phase 2 D-Window', 'Phase 2 D-Variance',
              'Phase 3 Window', 'Phase 3 Variance')
    hatches = ['/', '\\', '//', 'x', '+']
    colors = ['b', 'r', 'g', 'c', 'm', 'orange', 'indigo', 'gold']
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


def findRelevantVariables(Beta, n=48, m=50):
    RelevantVar = [[] for x in range(0, m)]
    for sensor in range(0, m):
        for (i, beta) in enumerate(Beta[sensor, 1:]):
            if abs(beta) > 0.0001:
                RelevantVar[sensor].append(i)
        log.add().debug('Relevant variables for sensor %d: %s' %
                        (sensor, RelevantVar[sensor]))
        log.sub()
    return RelevantVar


def regressionErrorLasso(model, X, y):
    Predict = model.predict(X)
    Error = np.subtract(Predict, y)
    return np.var(Error)


def numberMatches(Error):
    cnt = 0
    for row in Error:
        for x in row:
            if x < 0.0001:
                cnt += 1
    return cnt


def learnModelHourStationary(train_data, n=48, m=50):
    """
    Beta = m * m matrix,
    Beta[i] = [beta, beta_1,...,beta_m] for sensor i]
    Variance = [m] list, conditional variance of X[i,j+1] given
        X[1,j], X[2,j],...,X[m,j]
    InitMean = m length list, initial mean
    InitVar = m length list, cond variance e.g. regression error
    """
    Beta = np.zeros((m, m + 1))
    Variance = np.zeros(m)
    InitMean = np.zeros(m)
    InitVar = np.zeros(m)
    three_day = n * 3
    log.add().info("Lasso regression with alpha = %.3f" % alpha)
    log.sub()
    for i in range(0, m):
        X = [train_data[:, j] for j in range(0, three_day-1)]
        y = train_data[i][1:]
        """no intercept in the model"""
        model = linear_model.Lasso(alpha=alpha, fit_intercept=True,
                                   copy_X=True, max_iter=10000,
                                   warm_start=False)
        model.fit(X, y)
        Beta[i, 0] = model.intercept_
        Beta[i, 1:] = model.coef_
        Variance[i] = regressionErrorLasso(model, X, y)
        log.add().debug('Parameter for sensor %d: %s, %.2f' %
                        (i, str(Beta[i, :]), Variance[i]))
        log.sub()

    """just use this function to see how many zeros in Beta"""
    findRelevantVariables(Beta, n, m)
    for i in range(0, m):
        InitMean[i] = np.mean(train_data[i, :])
        InitVar[i] = np.var(train_data[i, :])

    log.add().debug('Init mean = %s' % str(InitMean))
    log.debug('Init variance = %s' % str(InitVar))
    log.debug("Cond variance = %s" % str(Variance))
    log.sub()
    return Beta, Variance, InitMean, InitVar


def windowInferHourStationary(Beta, CondVar, InitMean, InitVar,
                              Test, budget, n=96, m=50):
    """
    Test = m by n, test data
    Beta = m by m, parameters
    """
    Prediction = np.zeros((m, n))
    Error = np.zeros((m, n))
    MarginalVar = np.zeros((m, n))
    for j in range(0, n):
        """wrap up window indices"""
        window_start = (j * budget) % m
        window_end = window_start + budget
        window_indices = []
        for k in range(window_start, window_end):
            index = k % m
            window_indices.append(index)

        if j == 0:
            Prediction[:, j] = InitMean
            MarginalVar[:, j] = InitVar
            for index in window_indices:
                Prediction[index, j] = Test[index, j]
                MarginalVar[index, j] = 0
        else:
            # predict unobserved data with backward inference
            Prediction[:, j], MarginalVar[:, j] = \
                backwardInference(window_indices, CondVar, Beta, Test[:, j],
                                  Prediction[:, j-1], MarginalVar[:, j-1])

        log.add().debug('Window prediction at %d = %s'
                        % (j, Prediction[:, j]))
        log.sub()
        Error[:, j] = Test[:, j] - Prediction[:, j]
        Error[:, j] = np.absolute(Error[:, j])

    avg_error = np.sum(Error) / (m * n)
    cnt = numberMatches(Error)
    log.add().debug("Window Infer #Matches = %d" % cnt)
    log.sub()

    f = open('w%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error, cnt


def varianceInferHourStationary(Beta, CondVar, InitMean, InitVar,
                                Test, budget, n=96, m=50):
    """
    Test = m * n, test data
    Beta = m by (m+1)
    CondVar = m length list
    InitMean and InitVar = m length list
    """
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
            max_indices = findLargestK(MarginalVar[:, j], budget)
            for index in max_indices:
                Prediction[index][j] = Test[index][j]
                MarginalVar[index][j] = 0
        else:
            """
            observed sensor is in max_indices,
            who has large prediction variance in the last prediction
            """
            Prediction[:, j], MarginalVar[:, j] = \
                backwardInference(max_indices, CondVar, Beta, Test[:, j],
                                  Prediction[:, j-1], MarginalVar[:, j-1])
            max_indices = findLargestK(MarginalVar[:, j], budget)

        log.add().debug('Variance prediction at %d = %s'
                        % (j, Prediction[:, j]))
        log.sub()
        Error[:, j] = Test[:, j] - Prediction[:, j]
        Error[:, j] = np.absolute(Error[:, j])

    avg_error = np.sum(Error) / (m * n)
    cnt = numberMatches(Error)
    log.add().debug("#Matches = %d" % cnt)
    log.sub()

    f = open('v%d.csv' % budget, 'wb')
    writer = csv.writer(f)
    writer.writerow(title)
    for i in range(0, m):
        row = np.insert(Prediction[i, :], 0, i)
        writer.writerow(row)

    return avg_error, cnt


def backwardInference(ob_indices, CondVar, Beta, Test,
                      LastPredict, LastMarginalVar, m=50):
    """
    ob_indices: indices of observed variables
    CondVar: m * 1 list, cond variance of ALL variables
    Beta: m * (m+1) transition model
    Observed: ALL test data used as observation
    LastPredict: m * 1, u_t, update it using Kalman Gain equation
    LastMarginalVar: m * m, sigma_t, update is using Kalman Gain equation
    Prediction: m * 1, prediction based on its parents and observations
    """
    # patch u_t and sigma_t with 1 and 0
    LastMarginalVar = np.diag(np.insert(LastMarginalVar, 0, 0.0))
    LastPredict = np.insert(LastPredict, 0, 1.0)
    I = np.identity(m + 1)
    Prediction = np.zeros(m)
    MarginalVar = np.zeros(m)
    o = len(ob_indices)
    O = np.zeros(o)
    H = np.zeros((o, m + 1))
    R = np.zeros((o, o))
    for (i, index) in enumerate(ob_indices):
        H[i, :] = Beta[index, :]
        R[i, i] = CondVar[index]
        O[i] = Test[index]

    # update last time prediction first
    HT = np.transpose(H)
    K = np.dot(LastMarginalVar, HT)
    HSHT = np.dot(np.dot(H, LastMarginalVar), HT)
    K = np.dot(K, np.linalg.inv(HSHT + R))
    LastPredict = LastPredict + np.dot(K, (O - np.dot(H, LastPredict)))
    LastMarginalVar = np.dot((I - np.dot(K, H)), LastMarginalVar)

    # use them to predict unobserved value
    for i in range(0, m):
        if i not in ob_indices:
            Prediction[i] = np.dot(Beta[i, :], LastPredict)
            BetaSquared = np.square(Beta[i, 1:])
            LMV = [LastMarginalVar[k, k] for k in range(1, m+1)]
            MarginalVar[i] = CondVar[i] + np.dot(BetaSquared, LMV)
        else:
            Prediction[i] = Test[i]
            MarginalVar[i] = 0

    return Prediction, MarginalVar


def hourStationary(train_data, test_data):
    win_errs = [0] * len(budget_cnts)
    var_errs = [0] * len(budget_cnts)
    win_match = [0] * len(budget_cnts)
    var_match = [0] * len(budget_cnts)
    log.add().info('Hour stationary assumption')
    B, V, IM, IV = learnModelHourStationary(train_data)
    for i in range(0, len(budget_cnts)):
        win_errs[i], win_match[i] = \
            windowInferHourStationary(B, V, IM, IV, test_data, budget_cnts[i])
        log.add().info('Avg Window error =\t%.4f with %d budget' %
                       (win_errs[i], budget_cnts[i]))
        log.sub()
        var_errs[i], var_match[i] = \
            varianceInferHourStationary(B, V, IM, IV, test_data, budget_cnts[i])
        log.add().info('Avg Variance error =\t%.4f with %d budget' %
                       (var_errs[i], budget_cnts[i]))
        log.sub()
    log.info("#Window match = \t%s" % win_match)
    log.info("#Variance match = \t%s" % var_match)
    log.sub()

    return win_errs, var_errs


def main(train_file, test_file):
    """train_data = m * (3 day) shape matrix"""
    train_data = readInData(train_file)
    """test_data = m * (2 day) shape matrix"""
    test_data = readInData(test_file)
    h_win_err, h_var_err = hourStationary(train_data, test_data)
    return h_win_err, h_var_err


if __name__ == '__main__':
    # lg.basicConfig(level=lg.DEBUG)
    lg.basicConfig(level=lg.INFO)
    log = IndentedLoggerAdapter(lg.getLogger(__name__))
    np.set_printoptions(precision=4)
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

    alpha = 0.08
    topic = 'temperature_back_infer'
    log.info('Processing %s' % topic)
    p3_h_win, p3_h_var = \
        main('intelTemperatureTrain.csv', 'intelTemperatureTest.csv')
    # copy results of phase 1 and phase 2
    p1_win = [1.167, 1.049, 0.938, 0.692, 0.597]
    p1_var = [1.167, 0.967, 0.810, 0.560, 0.435]
    p2_h_win = [2.366, 1.561, 0.905, 0.412, 0.274]
    p2_h_var = [2.366, 1.519, 0.968, 0.454, 0.325]
    p2_d_win = [1.125, 0.94, 0.556, 0.279, 0.214]
    p2_d_var = [1.125, 0.774, 0.567, 0.295, 0.226]
    p3_d_win = [0, 0, 0, 0, 0]
    p3_d_var = [0, 0, 0, 0, 0]
    plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var, p2_d_win, p2_d_var,
                 p3_h_win, p3_h_var, y_max=3.5)

    alpha = 0.2
    topic = 'humidity_back_infer'
    log.info('Processing %s' % topic)
    p3_h_win, p3_h_var = \
        main('intelHumidityTrain.csv', 'intelHumidityTest.csv')
    # copy results of phase 1 and phase 2
    p1_win = [3.470, 3.119, 2.782, 2.070, 1.757]
    p1_var = [3.470, 3.160, 2.847, 2.172, 1.822]
    p2_h_win = [5.365, 2.701, 1.524, 0.689, 0.452]
    p2_h_var = [5.365, 2.964, 1.752, 0.786, 0.573]
    p2_d_win = [3.872, 2.327, 1.282, 0.718, 0.515]
    p2_d_var = [3.872, 2.123, 1.476, 0.744, 0.572]
    plotAvgError(p1_win, p1_var,
                 p2_h_win, p2_h_var, p2_d_win, p2_d_var,
                 p3_h_win, p3_h_var, y_max=8)
