from glob import glob
import pickle
import numpy as np
import scipy.stats
import math
import pylab
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, fbeta_score
import matplotlib.pyplot as plt

from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.stats import chi2, beta
from sklearn import svm
from sklearn.covariance import EllipticEnvelope

from dataCombiner import DataCombiner
from dataGenerator import DataGenerator
from numpy.linalg import inv, pinv


def addOneDayDataCode(lightAvg1, lightStd1, soundAvg1, soundStd1, screen1, move1,
        lightAvg2, lightStd2, soundAvg2, soundStd2, screen2, move2,
        light1AvgAvgList, light1AvgStdList, light1StdAvgList, light1StdStdList,
        sound1AvgAvgList, sound1AvgStdList, sound1StdAvgList, sound1StdStdList,
        screen1AvgList, screen1StdList, move1AvgList, move1StdList,
        light2AvgAvgList, light2AvgStdList, light2StdAvgList, light2StdStdList,
        sound2AvgAvgList, sound2AvgStdList, sound2StdAvgList, sound2StdStdList,
        screen2AvgList, screen2StdList, move2AvgList, move2StdList):
	# Add statistic values into lists for evening
    if len(lightAvg1) > 0:
        light1AvgAvgList.append(np.array(lightAvg1).mean())
        light1AvgStdList.append(np.array(lightAvg1).std())
        light1StdAvgList.append(np.array(lightStd1).mean())
        light1StdStdList.append(np.array(lightStd1).std())
        sound1AvgAvgList.append(np.array(soundAvg1).mean())
        sound1AvgStdList.append(np.array(soundAvg1).std())
        sound1StdAvgList.append(np.array(soundStd1).mean())
        sound1StdStdList.append(np.array(soundStd1).std())
        screen1AvgList.append(np.array(screen1).mean())
        screen1StdList.append(np.array(screen1).std())
        move1AvgList.append(np.array(move1).mean())
        move1StdList.append(np.array(move1).std())
    else:
        light1AvgAvgList.append(30)
        light1AvgStdList.append(30)
        light1StdAvgList.append(10)
        light1StdStdList.append(10)
        sound1AvgAvgList.append(35)
        sound1AvgStdList.append(10)
        sound1StdAvgList.append(30)
        sound1StdStdList.append(10)
        screen1AvgList.append(0)
        screen1StdList.append(0)
        move1AvgList.append(0)
        move1StdList.append(0)

    # Add statistic values into lists for morning
    if len(lightAvg2) > 0:
        light2AvgAvgList.append(np.array(lightAvg2).mean())
        light2AvgStdList.append(np.array(lightAvg2).std())
        light2StdAvgList.append(np.array(lightStd2).mean())
        light2StdStdList.append(np.array(lightStd2).std())
        sound2AvgAvgList.append(np.array(soundAvg2).mean())
        sound2AvgStdList.append(np.array(soundAvg2).std())
        sound2StdAvgList.append(np.array(soundStd2).mean())
        sound2StdStdList.append(np.array(soundStd2).std())
        screen2AvgList.append(np.array(screen2).mean())
        screen2StdList.append(np.array(screen2).std())
        move2AvgList.append(np.array(move2).mean())
        move2StdList.append(np.array(move2).std())
    else:
        light2AvgAvgList.append(30)
        light2AvgStdList.append(30)
        light2StdAvgList.append(10)
        light2StdStdList.append(10)
        sound2AvgAvgList.append(35)
        sound2AvgStdList.append(10)
        sound2StdAvgList.append(30)
        sound2StdStdList.append(10)
        screen2AvgList.append(0)
        screen2StdList.append(0)
        move2AvgList.append(0)
        move2StdList.append(0)


def addContextData(lightAvg, lightStd, soundAvg, soundStd, screen, move, sensing):
    lightAvg.append(float(sensing[4]))
    lightStd.append(float(sensing[6]))
    soundAvg.append(float(sensing[8]))
    soundStd.append(float(sensing[10]))
    screen.append(float(sensing[15]))
    move.append(float(sensing[3]))


def getDataFromDbBySeg2(dbName, contextIdxList):
    combiner = DataCombiner(dbName)
    combiner.combineData()
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
    sensingData, label = generator.generateFullDataset(t = 12)
    sleepData = combiner.sleepData
    timeList = generator.fullCreateTimeList
    del combiner
    del generator
    del label

    AMatrix = list()
    hasData = False
    # a 24-dimension data set
    light1AvgAvgList = list()
    light1AvgStdList = list()
    light1StdAvgList = list()
    light1StdStdList = list()
    sound1AvgAvgList = list()
    sound1AvgStdList = list()
    sound1StdAvgList = list()
    sound1StdStdList = list()
    screen1AvgList = list()
    screen1StdList = list()
    move1AvgList = list()
    move1StdList = list()

    light2AvgAvgList = list()
    light2AvgStdList = list()
    light2StdAvgList = list()
    light2StdStdList = list()
    sound2AvgAvgList = list()
    sound2AvgStdList = list()
    sound2StdAvgList = list()
    sound2StdStdList = list()
    screen2AvgList = list()
    screen2StdList = list()
    move2AvgList = list()
    move2StdList = list()

    lightAvg1 = list()
    lightStd1 = list()
    soundAvg1 = list()
    soundStd1 = list()
    screen1 = list()
    move1 = list()
    lightAvg2 = list()
    lightStd2 = list()
    soundAvg2 = list()
    soundStd2 = list()
    screen2 = list()
    move2 = list()

    for i in range(1, len(sensingData)):
        sensing = sensingData[i]
        hour = sensing[2]

        if (timeList[i] - timeList[i-1]).total_seconds() >= 12 * 3600:
            if hasData:
                addOneDayDataCode(lightAvg1, lightStd1, soundAvg1, soundStd1, screen1, move1,
                    lightAvg2, lightStd2, soundAvg2, soundStd2, screen2, move2,
                    light1AvgAvgList, light1AvgStdList, light1StdAvgList, light1StdStdList,
                    sound1AvgAvgList, sound1AvgStdList, sound1StdAvgList, sound1StdStdList,
                    screen1AvgList, screen1StdList, move1AvgList, move1StdList,
                    light2AvgAvgList, light2AvgStdList, light2StdAvgList, light2StdStdList,
                    sound2AvgAvgList, sound2AvgStdList, sound2StdAvgList, sound2StdStdList,
                    screen2AvgList, screen2StdList, move2AvgList, move2StdList)
                hasData = False

        if hour >= 22 or hour <= 0:
            addContextData(lightAvg1, lightStd1, soundAvg1, soundStd1, screen1, move1, sensing)
            hasData = True
        elif 7 <= hour < 10:
            addContextData(lightAvg2, lightStd2, soundAvg2, soundStd2, screen2, move2, sensing)
            hasData = True
        elif hour >= 10:
            if hasData:
                addOneDayDataCode(lightAvg1, lightStd1, soundAvg1, soundStd1, screen1, move1,
                    lightAvg2, lightStd2, soundAvg2, soundStd2, screen2, move2,
                    light1AvgAvgList, light1AvgStdList, light1StdAvgList, light1StdStdList,
                    sound1AvgAvgList, sound1AvgStdList, sound1StdAvgList, sound1StdStdList,
                    screen1AvgList, screen1StdList, move1AvgList, move1StdList,
                    light2AvgAvgList, light2AvgStdList, light2StdAvgList, light2StdStdList,
                    sound2AvgAvgList, sound2AvgStdList, sound2StdAvgList, sound2StdStdList,
                    screen2AvgList, screen2StdList, move2AvgList, move2StdList)
                hasData = False

    if hasData:
        addOneDayDataCode(lightAvg1, lightStd1, soundAvg1, soundStd1, screen1, move1,
            lightAvg2, lightStd2, soundAvg2, soundStd2, screen2, move2,
            light1AvgAvgList, light1AvgStdList, light1StdAvgList, light1StdStdList,
            sound1AvgAvgList, sound1AvgStdList, sound1StdAvgList, sound1StdStdList,
            screen1AvgList, screen1StdList, move1AvgList, move1StdList,
            light2AvgAvgList, light2AvgStdList, light2StdAvgList, light2StdStdList,
            sound2AvgAvgList, sound2AvgStdList, sound2StdAvgList, sound2StdStdList,
            screen2AvgList, screen2StdList, move2AvgList, move2StdList)

    AMatrix.append(light1AvgAvgList)
    AMatrix.append(light1AvgStdList)
    AMatrix.append(light1StdAvgList)
    AMatrix.append(light1StdStdList)
    AMatrix.append(sound1AvgAvgList)
    AMatrix.append(sound1AvgStdList)
    AMatrix.append(sound1StdAvgList)
    AMatrix.append(sound1StdStdList)
    AMatrix.append(screen1AvgList)
    AMatrix.append(screen1StdList)
    AMatrix.append(move1AvgList)
    AMatrix.append(move1StdList)

    AMatrix.append(light2AvgAvgList)
    AMatrix.append(light2AvgStdList)
    AMatrix.append(light2StdAvgList)
    AMatrix.append(light2StdStdList)
    AMatrix.append(sound2AvgAvgList)
    AMatrix.append(sound2AvgStdList)
    AMatrix.append(sound2StdAvgList)
    AMatrix.append(sound2StdStdList)
    AMatrix.append(screen2AvgList)
    AMatrix.append(screen2StdList)
    AMatrix.append(move2AvgList)
    AMatrix.append(move2StdList)

    sleeps = dict()
    sleepTimeList = list()
    wakeupTimeList = list()
    durationList = list()
    midList = list()
    for j in range(len(sleepData)):
        sleepLog = sleepData[j]
        sleepTime = sleepLog.sleepTime
        wakeupTime = sleepLog.wakeupTime
        duration = (wakeupTime - sleepTime).total_seconds() / 60
        #print "SleepTime: " + str(sleepTime) + ", WakeupTime: " + str(wakeupTime) + ", Duration: " + str(duration)
        if sleepTime.hour <= 12:
            sleepValue = (sleepTime.hour + 24) * 60 + sleepTime.minute
        else:
            sleepValue = sleepTime.hour * 60 + sleepTime.minute
        wakeupValue = wakeupTime.hour * 60 + wakeupTime.minute
        sleepTimeList.append(sleepValue)
        wakeupTimeList.append(wakeupValue)
        durationList.append(duration)
        midList.append(sleepValue + duration / 2)
        #print "Sleep = %d, Wake = %d, Duration = %d, Mid = %d\n" % (sleepValue, wakeupValue, duration, sleepValue + duration / 2)
    sleeps['sleep'] = sleepTimeList
    sleeps['wakeup'] = wakeupTimeList
    sleeps['duration'] = durationList
    sleeps['mid'] = midList
    return np.array(AMatrix), sleeps

def getDataFromDbBySeg(dbName, contextIdxList1, contextIdxList2):
    combiner = DataCombiner(dbName)
    combiner.combineData()
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
    sensingData, label = generator.generateFullDataset(t = 12)
    sleepData = combiner.sleepData
    timeList = generator.fullCreateTimeList
    del combiner
    del generator
    del label

    # The matrix of profiling, need to return
    AMatrix = list()
    contextNum1 = len(contextIdxList1)
    contextNum2 = len(contextIdxList2)
    oneDayList = [0] * (contextNum1 * 2 + contextNum2 *2)
    counts = [0] * (contextNum1 * 2 + contextNum2 *2)
    hasData = False

    for i in range(1, len(sensingData)):
        sensing = sensingData[i]
        hour = sensing[2]

        # Two consecutive sleeplogs are not in two consecutive days
        if (timeList[i] - timeList[i-1]).total_seconds() >= 12 * 3600:
            if hasData:
                for k in range(len(counts)):
                    if counts[k] != 0:
                        oneDayList[k] /= float(counts[k])
                AMatrix.append(oneDayList)
                oneDayList = [0] * (contextNum1 * 2 + contextNum2 *2)
                counts = [0] * (contextNum1 * 2 + contextNum2 *2)
                hasData = False

        """
        if hour >= 21 or hour < 1:
            for k in range(contextNum1):
                #oneDayList[k] = max(oneDayList[k], sensing[contextIdxList1[k]])
                oneDayList[k] += sensing[contextIdxList1[k]]
                counts[k] += 1
            hasData = True
        elif 7 <= hour < 11:
            for k in range(contextNum1):
                #oneDayList[contextNum + k] = max(oneDayList[contextNum + k], sensing[contextIdxList1[k]])
                oneDayList[contextNum1 + k] += sensing[contextIdxList1[k]]
                counts[contextNum1 + k] += 1
            hasData = True
        elif hour >= 11:
            if hasData:
                for idx in range(len(counts)):
                    if counts[idx] != 0:
                        oneDayList[idx] /= float(counts[idx])
                AMatrix.append(oneDayList)
                oneDayList = [0] * contextNum1 * 2
                counts = [0] * contextNum1 * 2
                hasData = False
        """


        if hour >= 22:
            for k in range(contextNum1):
                #oneDayList[k] = max(oneDayList[k], sensing[contextIdxList[k]])
                oneDayList[k] += sensing[contextIdxList1[k]]
                counts[k] += 1
            hasData = True
        elif hour < 2:
            for k in range(contextNum2):
                #oneDayList[contextNum + k] = max(oneDayList[contextNum + k], sensing[contextIdxList[k]])
                oneDayList[contextNum1 + k] += sensing[contextIdxList2[k]]
                counts[contextNum1 + k] += 1
            hasData = True
        elif 7 <= hour < 9:
            for k in range(contextNum2):
                #oneDayList[contextNum + k] = max(oneDayList[contextNum + k], sensing[contextIdxList[k]])
                oneDayList[contextNum1 + contextNum2 + k] += sensing[contextIdxList2[k]]
                counts[contextNum1 + contextNum2 + k] += 1
            hasData = True
        elif 9 <= hour < 11:
            for k in range(contextNum1):
                #oneDayList[contextNum + k] = max(oneDayList[contextNum + k], sensing[contextIdxList[k]])
                oneDayList[contextNum2 * 2 + contextNum1 + k] += sensing[contextIdxList1[k]]
                counts[contextNum2 * 2 + contextNum1 + k] += 1
            hasData = True
        elif hour >= 11:
            if hasData:
                for k in range(len(counts)):
                    if counts[k] != 0:
                        oneDayList[k] /= float(counts[k])
                AMatrix.append(oneDayList)
                oneDayList = [0] * (contextNum1 * 2 + contextNum2 *2)
                counts = [0] * (contextNum1 * 2 + contextNum2 *2)
                hasData = False


    if hasData:
        for k in range(len(counts)):
            if counts[k] != 0:
                oneDayList[k] /= float(counts[k])
        AMatrix.append(oneDayList)

    sleeps = dict()
    sleepTimeList = list()
    wakeupTimeList = list()
    durationList = list()
    midList = list()
    for j in range(len(sleepData)):
        sleepLog = sleepData[j]
        sleepTime = sleepLog.sleepTime
        wakeupTime = sleepLog.wakeupTime
        duration = (wakeupTime - sleepTime).total_seconds() / 60
        #print "SleepTime: " + str(sleepTime) + ", WakeupTime: " + str(wakeupTime) + ", Duration: " + str(duration)
        if sleepTime.hour <= 12:
            sleepValue = (sleepTime.hour + 24) * 60 + sleepTime.minute
        else:
            sleepValue = sleepTime.hour * 60 + sleepTime.minute
        wakeupValue = wakeupTime.hour * 60 + wakeupTime.minute
        sleepTimeList.append(sleepValue)
        wakeupTimeList.append(wakeupValue)
        durationList.append(duration)
        midList.append(sleepValue + duration / 2)
        #print "Sleep = %d, Wake = %d, Duration = %d, Mid = %d\n" % (sleepValue, wakeupValue, duration, sleepValue + duration / 2)
    sleeps['sleep'] = sleepTimeList
    sleeps['wakeup'] = wakeupTimeList
    sleeps['duration'] = durationList
    sleeps['mid'] = midList
    return np.array(AMatrix).T, sleeps


def getDataFromDbByHour(dbName, contextIdxList, hourList):
    combiner = DataCombiner(dbName)
    combiner.combineData()
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
    sensingData, label = generator.generateFullDataset(t = 12)
    sleepData = combiner.sleepData
    timeList = generator.fullCreateTimeList
    del combiner
    del generator
    del label

    # The matrix of profiling, need to return
    AMatrix = list()
    hourNum = len(hourList)
    contextNum = len(contextIdxList)
    oneDayList = [0] * contextNum * hourNum
    counts = [0] * contextNum * hourNum
    hasData = False

    for i in range(1, len(sensingData)):
        sensing = sensingData[i]
        hour = sensing[2]

        # Two consecutive sleeplogs are not in two consecutive days
        if (timeList[i] - timeList[i-1]).total_seconds() >= 12 * 3600:
            if hasData:
                #for idx in range(len(counts)):
                #    if counts[idx] != 0:
                #        oneDayList[idx] /= counts[idx]
                AMatrix.append(oneDayList)
                oneDayList = [0] * contextNum * hourNum
                counts = [0] * contextNum * hourNum
                hasData = False

        # Search the right hour bucket
        for j in range(len(hourList)):
            if hourList[j] == hour:
                for k in range(len(contextIdxList)):
                    oneDayList[j * contextNum + k] += sensing[contextIdxList[k]]
                    #oneDayList[j * contextNum + k] = max(oneDayList[j * contextNum + k], sensing[contextIdxList[k]])
                    counts[j * contextNum + k] += 1
                hasData = True
                break

        # Finished one day data, the time is after 10AM
        if hourList[-1] < hour < hourList[0]:
            if hasData:
                #for idx in range(len(counts)):
                #    if counts[idx] != 0:
                #        oneDayList[idx] /= counts[idx]
                AMatrix.append(oneDayList)
                oneDayList = [0] * contextNum * hourNum
                counts = [0] * contextNum * hourNum
                hasData = False
    if hasData:
        #for idx in range(len(counts)):
        #    if counts[idx] != 0:
        #        oneDayList[idx] /= counts[idx]
        AMatrix.append(oneDayList)

    sleeps = dict()
    sleepTimeList = list()
    wakeupTimeList = list()
    durationList = list()
    midList = list()
    for i in range(len(sleepData)):
        sleepLog = sleepData[i]
        sleepTime = sleepLog.sleepTime
        wakeupTime = sleepLog.wakeupTime
        duration = (wakeupTime - sleepTime).total_seconds() / 60
        #print "SleepTime: " + str(sleepTime) + ", WakeupTime: " + str(wakeupTime) + ", Duration: " + str(duration)
        if sleepTime.hour <= 12:
            sleepValue = (sleepTime.hour + 24) * 60 + sleepTime.minute
        else:
            sleepValue = sleepTime.hour * 60 + sleepTime.minute
        wakeupValue = wakeupTime.hour * 60 + wakeupTime.minute
        sleepTimeList.append(sleepValue)
        wakeupTimeList.append(wakeupValue)
        durationList.append(duration)
        midList.append(sleepValue + duration / 2)
        #print "Sleep = %d, Wake = %d, Duration = %d, Mid = %d\n" % (sleepValue, wakeupValue, duration, sleepValue + duration / 2)
    sleeps['sleep'] = sleepTimeList
    sleeps['wakeup'] = wakeupTimeList
    sleeps['duration'] = durationList
    sleeps['mid'] = midList
    return np.array(AMatrix).T, sleeps



def getMahalanobisDistance(x, sigma, mu):
    xarr = np.matrix(x)
    muarr = np.matrix(mu)
    ret = math.sqrt((xarr - muarr) * pinv(np.array(sigma)) * (xarr.T - muarr.T))
    return ret


def getAllUsersDataByHour(contextIdxList, hourList):
    allDbList = glob('./data/*.db')
    #allDbList = ["./data/6e44881f5af5d54a452b99f57899a7.db"]

    retValues = dict()
    for dbName in allDbList:
        oneUser = dict()
        AMatrix, sleeps = getDataFromDbByHour(dbName, contextIdxList, hourList)
        oneUser['data'] = AMatrix
        oneUser['sleep'] = sleeps
        retValues[dbName] = oneUser
    with open("values/profilingData.pickle", 'w') as f:
        pickle.dump(retValues, f)

def getAllUsersDataBySeg(contextIdxList1, contextIdxList2):
    allDbList = glob('./data/*.db')

    retValues = dict()
    for dbName in allDbList:
        oneUser = dict()
        AMatrix, sleeps = getDataFromDbBySeg(dbName, contextIdxList1, contextIdxList2)
        #AMatrix, sleeps = getDataFromDbBySeg2(dbName, contextIdxList1)
        oneUser['data'] = AMatrix
        oneUser['sleep'] = sleeps
        retValues[dbName] = oneUser
    with open("values/profilingData.pickle", 'w') as f:
        pickle.dump(retValues, f)

def getOutlierThreshold(AMatrix, sigma, mu, percent):
    d = list()
    arr = np.array(AMatrix)
    days = arr.shape[1]
    for i in range(days):
        x = list(arr[:, i])
        distance = getMahalanobisDistance(x, sigma, mu)
        d.append(distance)
    arrD = np.array(d)
    muD = arrD.mean()
    sigmaD = arrD.std()

    upper = muD + sigmaD * scipy.stats.norm.ppf(percent)
    return upper, upper, upper, upper
    """
    cdfValue1 = 2 * scipy.stats.norm.cdf(ratios['sleep']) - 1
    newRatio1 = scipy.stats.norm.ppf(cdfValue1)
    upper1 = muD + newRatio1 * sigmaD

    cdfValue2 = 2 * scipy.stats.norm.cdf(ratios['wakeup']) - 1
    newRatio2 = scipy.stats.norm.ppf(cdfValue2)
    upper2 = muD + newRatio2 * sigmaD

    cdfValue3 = 2 * scipy.stats.norm.cdf(ratios['duration']) - 1
    newRatio3 = scipy.stats.norm.ppf(cdfValue3)
    upper3 = muD + newRatio3 * sigmaD

    cdfValue4 = 2 * scipy.stats.norm.cdf(ratios['mid']) - 1
    newRatio4 = scipy.stats.norm.ppf(cdfValue4)
    upper4 = muD + newRatio4 * sigmaD
    return upper1, upper2, upper3, upper4
    """

def leaveOneOutTest(AMatrix, percent):
    ret = dict()
    sleepRet = list()
    wakeupRet = list()
    durationRet = list()
    midRet = list()
    distList = list()

    arrA = np.array(AMatrix)
    featureNo, days = arrA.shape
    for i in range(days):
        x = list(arrA[:,i])
        trainA = list()
        mu = list()
        for j in range(featureNo):
            Fj = list(arrA[j,:])
            Fj.pop(i)
            mu.append(np.array(Fj).mean())
            trainA.append(Fj)
        sigma = np.cov(trainA)
        #upper1, upper2, upper3, upper4 = getOutlierThreshold(trainA, sigma, mu, percent)
        dis = getMahalanobisDistance(x, sigma, mu)
        #distList.append(np.power(dis, 0.25))
        distList.append(dis)

    distMu = np.array(distList).mean()
    distSigma = np.array(distList).std()
    upper = distMu + distSigma * scipy.stats.norm.ppf(percent)
    for i in range(days):
        if distList[i] <= upper:
            sleepRet.append(0)
        else:
            sleepRet.append(1)
        if distList[i] <= upper:
            wakeupRet.append(0)
        else:
            wakeupRet.append(1)
        if distList[i] <= upper:
            durationRet.append(0)
        else:
            durationRet.append(1)
        if distList[i] <= upper:
            midRet.append(0)
        else:
            midRet.append(1)
    ret['sleep'] = sleepRet
    ret['wakeup'] = wakeupRet
    ret['duration'] = durationRet
    ret['mid'] = midRet
    ret['distances'] = distList
    return ret

def getAdvancedMadFiltering(raw, cutOff):
    c = 1.1926
    b = 1.4826
    ret = list()


    outer = list()
    for i in range(len(raw)):
        inner = list()
        for j in range(len(raw)):
            inner.append(np.absolute(raw[i] - raw[j]))
        inner_median = np.median(inner)
        outer.append(inner_median)
    outer_median = np.median(outer)
    Sn = c * outer_median
    #cutOff = float(cutOff) / Sn
    print "#####################  Sn = %f" % Sn

    for i in range(len(raw)):
        tmp = list()
        for j in range(len(raw)):
            tmp.append(np.absolute(raw[i] - raw[j]))
        median = np.median(tmp)
        if median / Sn >= cutOff:
            ret.append(1)
        else:
            ret.append(0)

    """
    median_of_all = np.median(raw)
    median_diffs = np.absolute(np.array(raw) - median_of_all)
    mad = b * np.median(median_diffs)

    for i in range(len(raw)):
        ratio = np.absolute(raw[i] - median_of_all) / mad
        if ratio >= cutOff:
            ret.append(1)
        else:
            ret.append(0)
    """

    return ret


def getLeaveOneOutGroundTruth(sleeps, needConfidence):

    ret = dict()
    total = list()

    cutOff = scipy.stats.norm.ppf((1.0+needConfidence)/2)
    #cutOff = 120
    sleep = getAdvancedMadFiltering(sleeps['sleep'], cutOff)
    wakeup = getAdvancedMadFiltering(sleeps['wakeup'], cutOff)
    duration = getAdvancedMadFiltering(sleeps['duration'], cutOff)
    mid = getAdvancedMadFiltering(sleeps['mid'], cutOff)
    for i in range(len(sleep)):
        if sleep[i] == 1 and mid[i] == 1:
            total.append(1)
        else:
            total.append(0)

    ret['sleep'] = sleep
    ret['wakeup'] = wakeup
    ret['duration'] = duration
    ret['mid'] = mid
    ret['total'] = total
    return ret



    """
    ret = dict()
    sleep = list()
    wakeup = list()
    duration = list()
    mid = list()
    total = list()

    sleepMean = np.array(sleeps['sleep']).mean()
    sleepStd = np.array(sleeps['sleep']).std()
    wakeupMean = np.array(sleeps['wakeup']).mean()
    wakeupStd = np.array(sleeps['wakeup']).std()
    durationMean = np.array(sleeps['duration']).mean()
    durationStd = np.array(sleeps['duration']).std()
    midMean = np.array(sleeps['mid']).mean()
    midStd = np.array(sleeps['mid']).std()

    #sleepRatio = float(delta) / sleepStd
    #if sleepRatio < 1.85:
    #    sleepRatio = 1.85
    #sleepDelta = sleepRatio * sleepStd
    sleepDelta = sleepStd * scipy.stats.norm.ppf((1.0+needConfidence)/2)
    #print "SleepRatio = " + str(sleepRatio)

    #wakeupRatio = float(delta) / wakeupStd
    #if wakeupRatio < 1.85:
    #    wakeupRatio = 1.85
    #wakeupDelta = wakeupRatio * wakeupStd
    wakeupDelta = wakeupStd * scipy.stats.norm.ppf((1.0+needConfidence)/2)
    #print "WakeupRatio = " + str(wakeupRatio)

    #durationRatio = float(delta) / durationStd
    #if durationRatio < 1.85:
    #    durationRatio = 1.85
    #durationDelta = durationRatio * durationStd
    durationDelta = durationStd * scipy.stats.norm.ppf((1.0+needConfidence)/2)
    #print "DurationRatio = " + str(durationRatio)

    #midRatio = float(delta) / midStd
    #if midRatio < 1.85:
    #    midRatio = 1.85
    #midDelta = midRatio * midStd
    midDelta = midStd * scipy.stats.norm.ppf((1.0+needConfidence)/2)
    #print "MidRatio = " + str(midRatio)

    for i in range(len(sleeps['sleep'])):
        sleepValue = sleeps['sleep'][i]
        if sleepMean - sleepDelta <= sleepValue <= sleepMean + sleepDelta:
            sleep.append(0)
        else:
            #print "Sleep time irregular: " + str(sleepValue) + ", index = " + str(i)
            sleep.append(1)

        wakeupValue = sleeps['wakeup'][i]
        if wakeupMean - wakeupDelta <= wakeupValue <= wakeupMean + wakeupDelta:
            wakeup.append(0)
        else:
            #print "Wakeup time irregular: " + str(wakeupValue)+ ", index = " + str(i)
            wakeup.append(1)

        durationValue = sleeps['duration'][i]
        if durationMean - durationDelta <= durationValue <= durationMean + durationDelta:
            duration.append(0)
        else:
            #print "Duration time irregular: " + str(durationValue)+ ", index = " + str(i)
            duration.append(1)

        midValue = sleeps['mid'][i]
        if midMean - midDelta <= midValue <= midMean + midDelta:
            mid.append(0)
        else:
            #print "Mid time irregular: " + str(midValue)+ ", index = " + str(i)
            mid.append(1)

        if sleep[-1] == 1 and mid[-1] == 1:
            total.append(1)
        else:
            total.append(0)

    ret['sleep'] = sleep
    ret['wakeup'] = wakeup
    ret['duration'] = duration
    ret['mid'] = mid
    ret['total'] = total
    return ret
    """



def loadData():
    with open("values/profilingData.pickle") as f:
        ret = pickle.load(f)
    return ret


def plotHistogram(data, bins):
    #newData = np.power(data, 0.5)
    n, bins, patches = pylab.hist(data, bins=bins, normed=1)
    #n, bins, patches = plt.hist(predicts['distances'], bins=15)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
    mu = np.array(data).mean()
    sigma = np.array(data).std()
    y = pylab.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'k--', linewidth=1.5)
    plt.show()

def plotContextHistogram(AMatrix, hourIdx, totalContext, contextIdxList):
    contextNum = len(contextIdxList)

    fig = plt.figure(figsize=(16, 6))
    plt.subplots_adjust(wspace=0.4,hspace=0.2)
    for i in range(contextNum):
        ax1 = fig.add_subplot(2, contextNum, i+1)
        data = list(AMatrix[hourIdx * totalContext + contextIdxList[i]])
        print "[Context %d] Max = %f, Min = %f" % (i, np.array(data).max(), np.array(data).min())
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax1.plot(bins, y, 'k--', linewidth=3.5)

    for i in range(contextNum):
        ax2 = fig.add_subplot(2, contextNum, contextNum + i + 1)
        data = np.power(AMatrix[hourIdx * totalContext + contextIdxList[i]], 1.0/8)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax2.plot(bins, y, 'k--', linewidth=3.5)
    plt.show()

def plotPerContextHistogram(AMatrix, totalContext, idx):
    fig = plt.figure(figsize=(12, 24))
    plt.subplots_adjust(wspace=0.4,hspace=0.2)
    for i in range(totalContext):
        ax1 = fig.add_subplot(7, totalContext, i+1)
        data = list(AMatrix[idx * totalContext + i])
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax1.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax2 = fig.add_subplot(7, totalContext, i+1+totalContext)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/4)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax2.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax3 = fig.add_subplot(7, totalContext,i+1+totalContext*2)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/5)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax3.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax4 = fig.add_subplot(7, totalContext, i+1+totalContext*3)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/6)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax4.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax5 = fig.add_subplot(7, totalContext, i+1+totalContext*4)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/7)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax5.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax6 = fig.add_subplot(7, totalContext, i+1+totalContext*5)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/8)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax6.plot(bins, y, 'k--', linewidth=3.5)
    for i in range(totalContext):
        ax7 = fig.add_subplot(7, totalContext, i+1+totalContext*6)
        data = np.power(AMatrix[idx * totalContext + i], 1.0/9)
        n, bins, patches = pylab.hist(data, bins=20, normed=1)
        plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
        mu = np.array(data).mean()
        sigma = np.array(data).std()
        y = pylab.normpdf(bins, mu, sigma)
        ax7.plot(bins, y, 'k--', linewidth=3.5)
    plt.show()


def getDistancesAndPredicts(raw, needConfidence):
    N, D = raw.shape
    predict = list()
    robust_cov = MinCovDet(support_fraction=0.85, random_state=0).fit(raw)
    robust_mahal = robust_cov.mahalanobis(raw)
    robust_mahal *= 1.0 * N / ((N - 1) ** 2)


    #upper = chi2.ppf(needConfidence, D)
    upper = beta.ppf(needConfidence, 1.0*D/2, 1.0*(N-D-1)/2)
    print "Upper = %f" % upper
    for i in range(len(robust_mahal)):
        if robust_mahal[i] > upper:
            predict.append(1)
        else:
            predict.append(0)

    #predict = getAdvancedMadFiltering(robust_mahal, 1.96)
    return predict, robust_mahal


def robustCovarianceEstimation(raw, needConfidence):
    ret = dict()
    D, N = raw.shape
    dataAll = raw.T
    dataSleep = raw[:D/2].T
    dataWakeup = raw[D/2:].T

    predict_all, distance_all = getDistancesAndPredicts(dataAll, needConfidence)
    ret['duration'] = predict_all
    ret['mid'] = predict_all
    ret['total'] = predict_all
    ret['distance_all'] = distance_all

    predict_sleep, distance_sleep = getDistancesAndPredicts(dataSleep, needConfidence)
    ret['sleep'] = predict_sleep
    ret['distance_sleep'] = distance_sleep

    predict_wakeup, distance_wakeup = getDistancesAndPredicts(dataWakeup, needConfidence)
    ret['wakeup'] = predict_wakeup
    ret['distance_wakeup'] = distance_wakeup
    return ret

def getOneClassSvmPredicts(raw):
    N, D = raw.shape
    predict = list()
    distances = list()
    for i in range(N):
        training = list(raw)
        x = training.pop(i)
        clf = svm.OneClassSVM(nu=0.05, kernel="rbf", gamma=0.1)
        clf.fit(np.array(training))
        predict.append(clf.predict(x))
        distances.append(clf.decision_function(x))

    for i in range(N):
        if predict[i] < 0:
            predict[i] = 1
        else:
            predict[i] = 0
    return predict, distances


def oneClassSvmEstimation(raw):
    ret = dict()
    D, N = raw.shape
    dataAll = raw.T
    dataSleep = raw[:D/2].T
    dataWakeup = raw[D/2:].T

    predict_all, distance_all = getOneClassSvmPredicts(dataAll)
    ret['duration'] = predict_all
    ret['mid'] = predict_all
    ret['total'] = predict_all
    ret['distance_all'] = distance_all

    predict_sleep, distance_sleep = getOneClassSvmPredicts(dataSleep)
    ret['sleep'] = predict_sleep
    ret['distance_sleep'] = distance_sleep

    predict_wakeup, distance_wakeup = getOneClassSvmPredicts(dataWakeup)
    ret['wakeup'] = predict_wakeup
    ret['distance_wakeup'] = distance_wakeup
    return ret



def main():
    runContextIdxList1 = [3, 4, 8, 15]
    runContextIdxList2 = [4, 8, 15]
    runHourList = [22, 23, 0, 1, 2, 6, 7, 8, 9, 10]
    #getAllUsersDataBySeg(runContextIdxList1, runContextIdxList2)
    #getAllUsersDataByHour(runContextIdxList2, runHourList)
    values = loadData()
    confidence = 0.95

    dbList = glob('./data/*.db')
    distDict = dict()
    for aDbName in dbList:
        # This user does not have light sensor, should be skipped
        if aDbName == "./data/af7e6c7446233beb982118d88c284768.db":
            continue
        A = values[aDbName]['data']
        if len(A[0]) <= 50:
            continue
        print "\n\n============================================================="
        print aDbName

        #plotHistogram(values[aDbName]['sleep']['mid'], 20)
        #plotContextHistogram(A, 3, len(runContextIdxList), [0,1,2,3])
        #plotPerContextHistogram(A, 7, 0)
        #continue

        runSleeps = values[aDbName]['sleep']
        groundTruth = getLeaveOneOutGroundTruth(runSleeps, confidence)
        #predicts = leaveOneOutTest(np.power(A, 0.25), confidence)
        predicts = robustCovarianceEstimation(np.power(A, 1.0/4), 0.99)
        #predicts = oneClassSvmEstimation(A)
        distDict[aDbName] = predicts

        #plotHistogram(predicts['distances_all'], 20)

        #for idx in range(len(predicts['sleep'])):
        #    if predicts['sleep'][idx] != groundTruth['sleep'][idx]:
        #        print "ID = %d, Predict = %d, Truth = %d, Distance = %f" % (idx, predicts['sleep'][idx], groundTruth['sleep'][idx], predicts['distance_sleep'][idx])

        print "\nTotal irregular sleep time detected: " + str(sum(predicts['sleep']))
        print "Total irregular wakeup time detected: " + str(sum(predicts['wakeup']))
        print "Total irregular duration time detected: " + str(sum(predicts['duration']))
        print "Total irregular mid time detected: " + str(sum(predicts['mid']))


        #sleepAcc = fbeta_score(groundTruth['sleep'], predicts, beta = 0.5)
        sleepAcc = f1_score(groundTruth['sleep'], predicts['sleep'])
        matrix = confusion_matrix(groundTruth['sleep'], predicts['sleep'])
        print "\nBedtime detection Confusion Matrix:"
        print matrix
        print "Total irregular bedtime: " + str(sum(groundTruth['sleep']))

        #wakeupAcc = fbeta_score(groundTruth['wakeup'], predicts, beta = 0.5)
        wakeupAcc = f1_score(groundTruth['wakeup'], predicts['wakeup'])
        matrix = confusion_matrix(groundTruth['wakeup'], predicts['wakeup'])
        print "\nWakeupTime detection Confusion Matrix:"
        print matrix
        print "Total irregular wakeup time: " + str(sum(groundTruth['wakeup']))

        #durationAcc = fbeta_score(groundTruth['duration'], predicts, beta = 0.5)
        durationAcc = f1_score(groundTruth['duration'], predicts['duration'])
        matrix = confusion_matrix(groundTruth['duration'], predicts['duration'])
        print "\nDuration detection Confusion Matrix:"
        print matrix
        print "Total irregular duration: " + str(sum(groundTruth['duration']))

        midAcc = f1_score(groundTruth['mid'], predicts['mid'])
        matrix = confusion_matrix(groundTruth['mid'], predicts['mid'])
        print "\nMid point detection Confusion Matrix:"
        print matrix
        print "Total irregular Mid point: " + str(sum(groundTruth['mid']))

        """
        totalAcc = f1_score(groundTruth['total'], predicts['sleep'])
        matrix = confusion_matrix(groundTruth['total'], predicts['sleep'])
        print "\nIrregular detection Confusion Matrix:"
        print matrix
        print "Total irregular: " + str(sum(groundTruth['total']))
        """

        print "\nSleep Acc: %f, Wakeup Acc: %f, Duration Acc: %f, Mid Acc: %f" % (sleepAcc, wakeupAcc, durationAcc, midAcc)

    with open("./values/distanceResults.pickle", 'w') as f:
        pickle.dump(distDict, f)

if __name__ == "__main__":
    main()


