from glob import glob
import datetime
from matplotlib import lines
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from dataCombiner import DataCombiner
from dataGenerator import DataGenerator
from generalModel import GeneralModel
from irregularSleepDetector import getLeaveOneOutGroundTruth
from irregularSleepDetector import robustCovarianceEstimation
from plotSet import processSleeps, getErrors, getSleepErrorFig
from sleepClassifiers import SleepClassifiers
import pickle
import pylab
import matplotlib.pyplot as plt


def getLeaveOneDayOutIndices(timeList):
    n = len(timeList)
    start = 0
    testIdx = list()
    trainIdx = list()
    addedNewDay = False
    for i in range(1, n):
        current = timeList[i]
        previous = timeList[i-1]
        if current.hour >= 15 or (current - previous).total_seconds() >= 3600 * 12:
            if current.hour >= 15 and i == 1:
                addedNewDay = True
                continue
            if not addedNewDay:
                testIdx.append(range(start, i))
                trainIdx.append(range(start) + range(i, n))
                start = i
                addedNewDay = True
        else:
            addedNewDay = False
    if not addedNewDay:
        testIdx.append(range(start, i))
        trainIdx.append(range(start) + range(i, n))
    return trainIdx, testIdx

def test(dbName):
    combiner = DataCombiner(dbName)
    combiner.combineData()
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
    sensingData, label = generator.generateFullDataset(t = 12)
    del sensingData
    del label
    timeList = generator.fullCreateTimeList
    trainIdx, testIdx = getLeaveOneDayOutIndices(timeList)
    if len(testIdx) == len(combiner.sleepData):
        print "$$$$$$$$$$$ OK $$$$$$$$$$$$$$$$$"
    else:
        print "*********** Bad ****************"
    for fold in testIdx:
        print str(timeList[fold[0]]) + " ---- " + str(timeList[fold[-1]])
    print "\n\n"

def getLeaveOneDayOutClassificationCdfFig():
    seeds12 = dict()
    seeds12["./data/ae26d65557cce4db276d35639649791.db"] = 7
    seeds12["./data/658ac828bdadbddaa909315ad80ac8.db"] = 20
    seeds12["./data/c75ec66c111373f533609c70b151a4f4.db"] = 55
    seeds12["./data/35b615fa9cc4fcceba44f76633178e3.db"] = 10
    seeds12["./data/441fb510333a8c3a4e43f6bde46d397.db"] = 3
    seeds12["./data/9eb047582abeadc143f6ab5c5f3d99f.db"] = 18
    seeds12["./data/65e1dbb96210264efe93260dbd4b73.db"] = 39
    seeds12["./data/9fbac69d7e3caf32badec66d14d6159.db"] = 27
    seeds12["./data/c6bd3bfcbfdd5f17fb9d23484b8ab95.db"] = 7
    seeds12["./data/18dcdfbc751064e9251fa718a9319fe6.db"] = 13
    seeds12["./data/168fbaf1e036cd9561c08746eb7287dd.db"] = 15
    seeds12["./data/4b5e9ead5cba4a4d92dcdaa95962952e.db"] = 42
    seeds12["./data/be884bbdfbae8d46b597a4f63c8d14.db"] = 57
    seeds12["./data/a0f0364632be365c7c5534f9bd896d.db"] = 46
    seeds12["./data/e47332db45a82c8fd78f7aad8658132.db"] = 27
    seeds12["./data/af7e6c7446233beb982118d88c284768.db"] = 1
    seeds12["./data/6e44881f5af5d54a452b99f57899a7.db"] = 13
    seeds12["./data/2cb5821ed7556c652217680baeed382.db"] = 48
    seeds12["./data/e7cbc87f7ef9dcada3431f435b4db9.db"] = 25

    dbList = glob('./data/*.db')
    sleepTime = list()
    wakeTime = list()
    duration = list()
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        print "==================================   " + dbName
        combiner = DataCombiner(dbName)
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
        data, label = generator.generateFullDataset(t = 12)
        classifier = SleepClassifiers(data, label, np.array([]), np.array([]), np.array([]), np.array([]))
        trainFolds, testFolds = getLeaveOneDayOutIndices(generator.fullCreateTimeList)
        d0Raw = combiner.getSleepLogFromTS(generator.fullCreateTimeList[0])
        d0 = list()
        for a in range(len(d0Raw)):
            item = d0Raw[a]
            if 120 <= item['duration'] <= 750:
                d0.append(item)
        print "===========  d0  ==========="
        for item in d0:
            print item
        print "\n"

        predicts = []
        for dayId in range(len(testFolds)):
            X_train, X_test = data[trainFolds[dayId]], data[testFolds[dayId]]
            Y_train, Y_test = label[trainFolds[dayId]], label[testFolds[dayId]]
            #clf = DecisionTreeClassifier(criterion = 'entropy')
            clf = RandomForestClassifier(n_estimators=35, criterion='entropy', n_jobs=-1, random_state=seeds12[dbName])
            clf.fit(X_train, Y_train)
            predicts += list(clf.predict(X_test))
            del X_train
            del X_test
            del Y_train
            del Y_test
            del clf
        smoothedPredicts = classifier.smoothPrediction(predicts, 0)
        d1Raw = classifier.getSleepTimeAndDuration(smoothedPredicts, generator.fullCreateTimeList)
        d1 = processSleeps(d1Raw, 120, 36)

        print "===========  d1  ==========="
        for item in d1:
            print item
        print "\nTruth: %d , predict: %d" % (len(d0), len(d1))

        getErrors(d0, d1, sleepTime, wakeTime, duration)
        del generator
        del combiner
        del data
        del label
        del d0Raw
        del d0
        del d1Raw
        del d1
    sleepTime = sorted(sleepTime)
    wakeTime = sorted(wakeTime)
    duration = sorted(duration)

    with open("values/leaveOneDayOutSleepErrors.pickle", 'w') as f:
        pickle.dump([sleepTime, wakeTime, duration], f)
    getSleepErrorFig("values/leaveOneDayOutSleepErrors.pickle")


def getNUSError():
    dbList = glob('./data/*.db')
    #dbList = ["./data/168fbaf1e036cd9561c08746eb7287dd.db"]
    sleepTime = list()
    wakeTime = list()
    duration = list()
    d0Mean = list()
    d0Std = list()
    d1Mean = list()
    d1Std = list()
    daysList = list()
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        print "==================================   " + dbName
        combiner = DataCombiner(dbName)
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
        data, label = generator.generateFullDataset(t = 12)
        d0Raw = combiner.getSleepLogFromTS(generator.fullCreateTimeList[0])
        d0 = list()
        for a in range(len(d0Raw)):
            item = d0Raw[a]
            if 120 <= item['duration'] <= 750:
                d0.append(item)
        daysList.append(len(d0))
        tmp = list()
        for a in range(len(d0)):
            tmp.append(d0[a]['duration'])
        d0Mean.append(np.array(tmp).mean())
        d0Std.append(np.array(tmp).std())
        del tmp

        d1 = list()
        oneRecord = dict()
        hasData = False
        for sid in range(1, len(generator.fullCreateTimeList)):
            ts = generator.fullCreateTimeList[sid]
            pre_ts = generator.fullCreateTimeList[sid-1]
            if 11 <= ts.hour <= 21:
                if not hasData:
                    continue
                else:
                    if oneRecord['duration'] >= 180:
                        d1.append(oneRecord)
                    oneRecord = dict()
                    oneRecord['start'] = ts
                    oneRecord['end'] = ts
                    oneRecord['duration'] = 0
                    hasData = False
            else:
                if (ts - pre_ts).total_seconds() >= 12 * 3600:
                    if hasData:
                        if oneRecord['duration'] >= 180:
                            d1.append(oneRecord)
                        oneRecord = dict()
                        oneRecord['start'] = ts
                        oneRecord['end'] = ts
                        oneRecord['duration'] = 0
                        hasData = True
                else:
                    if not hasData:
                        oneRecord['start'] = ts
                        oneRecord['end'] = ts
                        oneRecord['duration'] = 0
                        hasData = True
                    if data[sid][15] <= 30:
                        oneRecord['end'] = ts
                        oneRecord['duration'] = ((oneRecord['end'] - oneRecord['start']).total_seconds()) / 60
                    else:
                        if oneRecord['duration'] >= 180:
                            d1.append(oneRecord)
                            oneRecord = dict()
                        oneRecord['start'] = ts
                        oneRecord['end'] = ts
                        oneRecord['duration'] = 0
                        hasData = True
        print "===========  d1  ==========="
        for item in d1:
            print item
        print "\nTruth: %d , predict: %d" % (len(d0), len(d1))

        tmp = list()
        for a in range(len(d1)):
            tmp.append(d1[a]['duration'])
        d1Mean.append(np.array(tmp).mean())
        d1Std.append(np.array(tmp).std())
        del tmp

        getErrors(d0, d1, sleepTime, wakeTime, duration)
        del generator
        del combiner
        del data
        del label
        del d0Raw
        del d0
        del d1
    sleepTime = sorted(sleepTime)
    wakeTime = sorted(wakeTime)
    duration = sorted(duration)

    with open("values/nusSleepErrors.pickle", 'w') as f:
        pickle.dump([sleepTime, wakeTime, duration], f)
    with open("values/compareNusDuration.pickle", 'w') as f:
        pickle.dump([daysList, d0Mean, d0Std, d1Mean, d1Std], f)
    getSleepErrorFig("values/nusSleepErrors.pickle")


def getCompareDurationOfNusFig(pickleFile):
    with open(pickleFile) as f:
        daysList, d0Mean, d0Std, d1Mean, d1Std = pickle.load(f)

    userId = 0
    userIdList = list()
    d0Means = ()
    d0Stds = ()
    d1Means = ()
    d1Stds = ()
    for idx in range(len(daysList)):
        userId += 1
        if daysList[idx] >= 50:
            userIdList.append(userId)
            d0Means += (d0Mean[idx], )
            d0Stds += (d0Std[idx], )
            d1Means += (d1Mean[idx], )
            d1Stds += (d1Std[idx], )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.hold(True)
    colorList = ['darkkhaki','royalblue', 'r', 'm', 'b', 'y', 'c', 'g']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    error_config = {'ecolor': 'black', 'elinewidth': 1.5}
    plt.bar(index, d0Means, bar_width, color = colorList[0],
                yerr = d0Stds, error_kw = error_config, label = "Logged Sleeps")
    plt.bar(index + bar_width, d1Means, bar_width, color = colorList[1],
                yerr = d1Stds, error_kw = error_config, label = "Longest Non-usage Time")
    plt.hold(False)
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Sleep Duration (minutes)', fontsize='x-large')
    plt.xlim(0, len(userIdList))
    plt.xticks(index + bar_width, userIdList)
    plt.ylim(200, 710)
    #plt.yticks(np.arange(0.40, 1.15, 0.1))
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    plt.legend(loc = 2)
    plt.show()


def getSleepStatistic():
    dbList = glob('./data/*.db')
    daysList = list()
    sleepMean = list()
    sleepStd = list()
    wakeMean = list()
    wakeStd = list()
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        print "==================================   " + dbName
        combiner = DataCombiner(dbName)
        sleeps = combiner.sleepData
        daysList.append(len(sleeps))
        l1 = list()
        l2 = list()
        for ptr in range(len(sleeps)):
            sleep = sleeps[ptr]
            if sleep.sleepTime.hour < 12:
                l1.append((sleep.sleepTime.hour + 24) * 60 + sleep.sleepTime.minute)
            else:
                l1.append(sleep.sleepTime.hour * 60 + sleep.sleepTime.minute)
            l2.append(sleep.wakeupTime.hour * 60 + sleep.wakeupTime.minute)
        sleepMean.append(np.array(l1).mean())
        sleepStd.append(np.array(l1).std())
        wakeMean.append(np.array(l2).mean())
        wakeStd.append(np.array(l2).std())
    with open("values/sleepStatistics.pickle", 'w') as f:
        pickle.dump([daysList, sleepMean, sleepStd, wakeMean, wakeStd], f)


def getSleepStatisticFig():
    with open("values/sleepStatistics.pickle") as f:
        daysList, sleepMean, sleepStd, wakeMean, wakeStd = pickle.load(f)

    userIdList = list()
    userId = 0
    v0 = ()
    v1 = ()
    v2 = ()
    v3 = ()
    for idx in range(len(daysList)):
        userId += 1
        if daysList[idx] >= 50:
            userIdList.append(userId)
            v0 += (sleepMean[idx], )
            v1 += (sleepStd[idx], )
            v2 += (wakeMean[idx], )
            v3 += (wakeStd[idx], )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['darkkhaki','royalblue']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    error_config = {'ecolor': 'black', 'elinewidth': 1.5}
    bar1 = ax.bar(index, v0, bar_width, color = colorList[0],
                yerr = v1, error_kw = error_config, label = "Bedtime")
    ax2 = ax.twinx()
    bar2 = ax2.bar(index + bar_width, v2, bar_width, color = colorList[1],
                yerr = v3, error_kw = error_config, label = "Waketime")
    plt.legend((bar1[0], bar2[0]), ('Bedtime', 'Waketime'), 0)
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Bedtime', fontsize=16)
    ax2.set_ylabel('Waketime', fontsize=16)
    plt.xlim(0, len(userIdList))
    plt.xticks(index + bar_width, userIdList)
    ax.set_ylim(1260, 1690, 60)
    ax.set_yticks(np.arange(1260, 1690, 60))
    ax.set_yticklabels(["09PM", "10PM", "11PM", "12AM", "01AM", "02AM", "03AM", "04AM"])
    ax2.set_ylim(360, 730, 60)
    ax2.set_yticks(np.arange(360, 730, 60))
    ax2.set_yticklabels(["06AM", "07AM", "08AM", "09AM", "10AM", "11AM", "12PM"])
    plt.tight_layout()
    plt.show()

def getScreenOnStatistic():
    dbList = glob('./data/*.db')
    daysList = list()
    values = list()
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        print "==================================   " + dbName
        combiner = DataCombiner(dbName)
        daysList.append(len(combiner.sleepData))
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
        data, label = generator.generateFullDataset(t = 12)
        tmp = list()
        for ptr in range(1, len(data)):
            ts = generator.fullCreateTimeList[ptr]
            if 11 <= ts.hour <= 21:
                continue
            else:
                tmp.append(data[ptr][15])
        values.append(tmp)
        del combiner
        del generator
        del data
        del label
    with open("values/screenOnStatistics.pickle", 'w') as f:
        pickle.dump([daysList, values], f)
    getScreenOnStatisticFig()

def getScreenOnStatisticFig():
    with open("values/screenOnStatistics.pickle") as f:
        daysList, values = pickle.load(f)

    userIdList = list()
    meanList = list()
    stdList = list()
    for userId in range(0, len(daysList)):
        if daysList[userId] < 50:
            continue
        userIdList.append(userId + 1)
        meanList.append(np.array(values[userId]).mean())
        stdList.append(np.array(values[userId]).std())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['darkkhaki','royalblue']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    ax.bar(index, meanList, bar_width, color = colorList[0], label = "Mean")
    ax.bar(index + bar_width, stdList, bar_width, color = colorList[1], label = "STD")
    #plt.legend((bar1[0], bar2[0]), ('Bedtime', 'Waketime'), 0)
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Screen-On Statistics (seconds)', fontsize=16)
    plt.xlim(0, len(userIdList))
    plt.xticks(index + bar_width, userIdList)
    plt.legend(loc='upper right', fontsize='x-large')
    plt.tight_layout()
    plt.show()


def getPredictionErrorVariance():
    seeds12 = dict()
    seeds12["./data/ae26d65557cce4db276d35639649791.db"] = 7
    seeds12["./data/658ac828bdadbddaa909315ad80ac8.db"] = 20
    seeds12["./data/c75ec66c111373f533609c70b151a4f4.db"] = 55
    seeds12["./data/35b615fa9cc4fcceba44f76633178e3.db"] = 10
    seeds12["./data/441fb510333a8c3a4e43f6bde46d397.db"] = 3
    seeds12["./data/9eb047582abeadc143f6ab5c5f3d99f.db"] = 18
    seeds12["./data/65e1dbb96210264efe93260dbd4b73.db"] = 39
    seeds12["./data/9fbac69d7e3caf32badec66d14d6159.db"] = 27
    seeds12["./data/c6bd3bfcbfdd5f17fb9d23484b8ab95.db"] = 7
    seeds12["./data/18dcdfbc751064e9251fa718a9319fe6.db"] = 13
    seeds12["./data/168fbaf1e036cd9561c08746eb7287dd.db"] = 15
    seeds12["./data/4b5e9ead5cba4a4d92dcdaa95962952e.db"] = 42
    seeds12["./data/be884bbdfbae8d46b597a4f63c8d14.db"] = 57
    seeds12["./data/a0f0364632be365c7c5534f9bd896d.db"] = 46
    seeds12["./data/e47332db45a82c8fd78f7aad8658132.db"] = 27
    seeds12["./data/af7e6c7446233beb982118d88c284768.db"] = 1
    seeds12["./data/6e44881f5af5d54a452b99f57899a7.db"] = 13
    seeds12["./data/2cb5821ed7556c652217680baeed382.db"] = 48
    seeds12["./data/e7cbc87f7ef9dcada3431f435b4db9.db"] = 25

    dbList = glob('./data/*.db')
    sleepTime = list()
    sleepVar = list()
    wakeTime = list()
    wakeVar = list()
    duration = list()
    durationVar = list()
    tmp_screen_on = list()
    total_screen_on = list()
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        print "==================================   " + dbName
        combiner = DataCombiner(dbName)
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
        data, label = generator.generateFullDataset(t = 12)
        classifier = SleepClassifiers(data, label, np.array([]), np.array([]), np.array([]), np.array([]))
        trainFolds, testFolds = getLeaveOneDayOutIndices(generator.fullCreateTimeList)
        d0Raw = combiner.getSleepLogFromTS(generator.fullCreateTimeList[0])
        d0 = list()

        for a in range(len(d0Raw)):
            item = d0Raw[a]
            if 120 <= item['duration'] <= 750:
                d0.append(item)

        predicts = []
        for dayId in range(len(testFolds)):
            X_train, X_test = data[trainFolds[dayId]], data[testFolds[dayId]]
            Y_train, Y_test = label[trainFolds[dayId]], label[testFolds[dayId]]
            screenOnTime = 0
            for recordId in range(len(X_test)):
                screenOnTime += X_test[recordId][15]
            tmp_screen_on.append(screenOnTime)
            #clf = DecisionTreeClassifier(criterion = 'entropy')
            clf = RandomForestClassifier(n_estimators=35, criterion='entropy', n_jobs=-1, random_state=seeds12[dbName])
            clf.fit(X_train, Y_train)
            predicts += list(clf.predict(X_test))
            del X_train
            del X_test
            del Y_train
            del Y_test
            del clf
        smoothedPredicts = classifier.smoothPrediction(predicts, 0)
        d1Raw = classifier.getSleepTimeAndDuration(smoothedPredicts, generator.fullCreateTimeList)
        d1 = processSleeps(d1Raw, 120, 36)

        truthId = 0
        predictId = 0
        d0Sleep = list()
        d0Wakup = list()
        d0Duration = list()
        while True:
            if predictId >= len(d1) or truthId >= len(d0):
                break
            trueSleepTime = d0[truthId]['start']
            predSleepTime = d1[predictId]['start']
            sleepDelta = abs((predSleepTime - trueSleepTime).total_seconds()) / 60
            if sleepDelta < 180:
                sleepTime.append(sleepDelta)
                wakeDelta = abs((d0[truthId]['end'] - d1[predictId]['end']).total_seconds()) / 60
                wakeTime.append(wakeDelta)
                durationDelta = abs(d0[truthId]['duration'] - d1[predictId]['duration'])
                total_screen_on.append(tmp_screen_on[truthId] / 60)
                if durationDelta > 75 or sleepDelta > 30:
                    print "BedtimeDelta: " + str(sleepDelta) + " DurDelta: " + str(durationDelta) + " TT: " + str(d0[truthId]['start']) + ", PT: " + str(d1[predictId]['start']) + ", TD: " + str(d0[truthId]['duration']) + ", PD: " + str(d1[predictId]['duration'])
                duration.append(durationDelta)

                if trueSleepTime.hour < 12:
                    d0Sleep.append((trueSleepTime.hour + 24) * 60 + trueSleepTime.minute)
                else:
                    d0Sleep.append(trueSleepTime.hour * 60 + trueSleepTime.minute)
                d0Wakup.append(d0[truthId]['end'].hour * 60 + d0[truthId]['end'].minute)
                d0Duration.append(d0[truthId]['duration'])
                truthId += 1
                predictId += 1
            elif d0[truthId]['start'] < d1[predictId]['start']:
                truthId += 1
            else:
                predictId += 1
        sleepVar += list(abs(np.array(d0Sleep) - np.array(d0Sleep).mean()))
        wakeVar += list(abs(np.array(d0Wakup) - np.array(d0Wakup).mean()))
        durationVar += list(np.array(d0Duration) - np.array(d0Duration).mean())
        del generator
        del combiner
        del data
        del label
        del d0Raw
        del d0
        del d1
    with open("./values/correlationVarianceErr.pickle", 'w') as f:
        pickle.dump([sleepTime, wakeTime, duration, sleepVar, wakeVar, durationVar, total_screen_on], f)

def getPredictionErrorVarianceFig(showType):
    with open("./values/correlationVarianceErr.pickle") as f:
        sleepTime, wakeTime, duration, sleepVar, wakeVar, durationVar, total_screen_on = pickle.load(f)

    ax = plt.figure().gca()
    if showType is "sleep":
        plt.scatter(sleepVar, sleepTime, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(sleepVar, (len(sleepVar),1)), sleepTime)
        miny = np.array(sleepVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(sleepVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(sleepVar).min(), np.array(sleepVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_xlabel("Logged Bedtime Variance Compared to Mean (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_ylabel("Predicted Bedtime Error (minute)", fontsize=16)
        ax.set_ylim(0, 180)
    elif showType is "wakeup":
        plt.scatter(wakeVar, wakeTime, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(wakeVar, (len(wakeVar),1)), wakeTime)
        miny = np.array(wakeVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(wakeVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(wakeVar).min(), np.array(wakeVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_xlabel("Logged Waketime Variance Compared to Mean (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_ylabel("Predicted Waketime Error (minute)", fontsize=16)
        ax.set_ylim(0, 180)
    else:
        plt.scatter(durationVar, duration, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(durationVar, (len(durationVar),1)), duration)
        miny = np.array(durationVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(durationVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(durationVar).min(), np.array(durationVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_xlabel("Logged Duration Variance Compared to Mean (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_ylabel("Predicted Duration Error (minute)", fontsize=16)
        ax.set_ylim(0, 280)
    plt.show()


def getScreenOnTimeAndLoggedVarianceFig(showType):
    with open("./values/correlationVarianceErr.pickle") as f:
        sleepTime, wakeTime, duration, sleepVar, wakeVar, durationVar, total_screen_on = pickle.load(f)

    ax = plt.figure().gca()
    if showType is "sleep":
        plt.scatter(total_screen_on, sleepTime, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(sleepVar, (len(sleepVar),1)), sleepTime)
        miny = np.array(sleepVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(sleepVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(sleepVar).min(), np.array(sleepVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_ylabel("Predicted Bedtime Error (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_xlabel("Total Screen-On Time (minutes)", fontsize=16)
        ax.set_ylim(0, 180)
    elif showType is "wakeup":
        plt.scatter(total_screen_on, wakeTime, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(wakeVar, (len(wakeVar),1)), wakeTime)
        miny = np.array(wakeVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(wakeVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(wakeVar).min(), np.array(wakeVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_ylabel("Predicted Waketime Error (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_xlabel("Total Screen-On Time (minutes)", fontsize=16)
        ax.set_ylim(0, 180)
    else:
        plt.scatter(total_screen_on, duration, axes=ax)
        model = LinearRegression()
        model.fit(np.reshape(durationVar, (len(durationVar),1)), duration)
        miny = np.array(durationVar).min() * model.coef_[0] + model.intercept_
        maxy = np.array(durationVar).max() * model.coef_[0] + model.intercept_
        ax.add_line(lines.Line2D([np.array(durationVar).min(), np.array(durationVar).max()], [miny, maxy], color='k', linewidth=3))
        ax.set_ylabel("Predicted Duration Error (minutes)", fontsize=16)
        ax.set_xlim(0, 280)
        ax.set_xlabel("Total Screen-On Time (minutes)", fontsize=16)
        ax.set_ylim(0, 280)
    plt.show()

def getDistanceSleepVarianceFig(showType):
    with open("./values/profilingData.pickle") as f:
        valueDict = pickle.load(f)

    with open("./values/distanceResults.pickle") as f:
        distDict = pickle.load(f)

    varList = list()
    distList = list()
    dbList = glob('./data/*.db')
    for dbName in dbList:
        sleepDict = valueDict[dbName]['sleep']
        if len(sleepDict['sleep']) < 50:
            continue
        if showType == "sleep":
            sleeptimeList = sleepDict['sleep']
        elif showType == "wakeup":
            sleeptimeList = sleepDict['wakeup']
        else:
            sleeptimeList = sleepDict['duration']
        #sleeptimeList = sleepDict['duration']
        sleeptimeDelta = list(abs(np.array(sleeptimeList) - np.array(sleeptimeList).mean()))
        for idx in range(len(distDict[dbName]['distance_all'])):
            if sleeptimeDelta[idx] >= 75:
                varList.append(sleeptimeDelta[idx])
                distList.append(distDict[dbName]['distance_all'][idx])
    ax = plt.figure().gca()
    plt.scatter(varList, distList, axes=ax)
    model = LinearRegression()
    model.fit(np.reshape(varList, (len(varList),1)), distList)
    miny = np.array(varList).min() * model.coef_[0] + model.intercept_
    maxy = np.array(varList).max() * model.coef_[0] + model.intercept_
    ax.add_line(lines.Line2D([np.array(varList).min(), np.array(varList).max()], [miny, maxy], color='k', linewidth=3))
    if showType == "sleep":
        ax.set_xlabel("Absolute Logged Bedtime Variance (>=75 minutes)", fontsize=16)
    elif showType == "wakeup":
        ax.set_xlabel("Absolute Logged Waketime Variance (>=75 minutes)", fontsize=16)
    else:
        ax.set_xlabel("Absolute Logged Duration Variance (>=75 minutes)", fontsize=16)
    ax.set_xlim(75, 250)
    plt.xticks(np.arange(75, 251, 30))
    ax.set_ylabel("Context Mahalanobis Distance", fontsize=16)
    ax.set_ylim(0, 25)
    plt.show()


def getIrregularSleepFig():
    with open("values/profilingData.pickle") as f:
        values = pickle.load(f)
    confidence = 0.95
    dbList = glob('./data/*.db')
    userIdList = list()
    userId = 0
    precisionList = list()
    recallList = list()
    for aDbName in dbList:
        userId += 1
        A = values[aDbName]['data']
        if len(A[0]) <= 50:
            continue
        print "\n\n============================================================="
        print aDbName
        userIdList.append(userId)
        runSleeps = values[aDbName]['sleep']
        groundTruth = getLeaveOneOutGroundTruth(runSleeps, confidence)
        predicts = robustCovarianceEstimation(np.power(A, 1.0/4), 0.99)
        precisionList.append(precision_score(groundTruth['sleep'], predicts['sleep']))
        recallList.append(recall_score(groundTruth['sleep'], predicts['sleep']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['darkkhaki','royalblue']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    ax.bar(index, precisionList, bar_width, color = colorList[0], label = "Precision")
    ax.bar(index + bar_width, recallList, bar_width, color = colorList[1], label = "Recall")
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    plt.xlim(0, len(userIdList))
    plt.xticks(index + bar_width, userIdList)
    plt.legend(loc='upper right', fontsize='x-large')
    plt.tight_layout()
    plt.show()


def getIrregularWakeupFig():
    with open("values/profilingData.pickle") as f:
        values = pickle.load(f)
    confidence = 0.95
    dbList = glob('./data/*.db')
    userIdList = list()
    userId = 0
    precisionList = list()
    recallList = list()
    for aDbName in dbList:
        userId += 1
        A = values[aDbName]['data']
        if len(A[0]) <= 50:
            continue
        print "\n\n============================================================="
        print aDbName
        userIdList.append(userId)
        runSleeps = values[aDbName]['sleep']
        groundTruth = getLeaveOneOutGroundTruth(runSleeps, confidence)
        predicts = robustCovarianceEstimation(np.power(A, 1.0/4), 0.99)
        precisionList.append(precision_score(groundTruth['wakeup'], predicts['wakeup']))
        recallList.append(recall_score(groundTruth['wakeup'], predicts['wakeup']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['darkkhaki','royalblue']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    ax.bar(index, precisionList, bar_width, color = colorList[0], label = "Precision")
    ax.bar(index + bar_width, recallList, bar_width, color = colorList[1], label = "Recall")
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    plt.xlim(0, len(userIdList))
    ax.set_ylim(0, 0.6)
    plt.xticks(index + bar_width, userIdList)
    plt.legend(loc='upper center', fontsize='x-large')
    plt.tight_layout()
    plt.show()


def getIrregularDurationFig():
    with open("values/profilingData.pickle") as f:
        values = pickle.load(f)
    confidence = 0.95
    dbList = glob('./data/*.db')
    userIdList = list()
    userId = 0
    precisionList = list()
    recallList = list()
    for aDbName in dbList:
        userId += 1
        A = values[aDbName]['data']
        if len(A[0]) <= 50:
            continue
        print "\n\n============================================================="
        print aDbName
        userIdList.append(userId)
        runSleeps = values[aDbName]['sleep']
        groundTruth = getLeaveOneOutGroundTruth(runSleeps, confidence)
        predicts = robustCovarianceEstimation(np.power(A, 1.0/4), 0.99)
        precisionList.append(precision_score(groundTruth['duration'], predicts['duration']))
        recallList.append(recall_score(groundTruth['duration'], predicts['duration']))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['darkkhaki','royalblue']
    margin = 0.25
    bar_width = (1.0 - margin) / 2
    index = np.arange(len(userIdList))
    ax.bar(index, precisionList, bar_width, color = colorList[0], label = "Precision")
    ax.bar(index + bar_width, recallList, bar_width, color = colorList[1], label = "Recall")
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Score', fontsize=16)
    ax.set_ylim(0, 1.0)
    plt.xlim(0, len(userIdList))
    plt.xticks(index + bar_width, userIdList)
    plt.legend(loc='upper right', fontsize='x-large')
    plt.tight_layout()
    plt.show()

def plotSleepHistogramFig():
    with open("values/profilingData.pickle") as f:
        ret = pickle.load(f)

    p1DbName = "./data/af7e6c7446233beb982118d88c284768.db"
    p2DbName = "./data/6e44881f5af5d54a452b99f57899a7.db"
    p1Sleeps = ret[p1DbName]['sleep']
    p2Sleeps = ret[p2DbName]['sleep']

    fig = plt.figure()
    plt.subplots_adjust(wspace=0.1,hspace=0.2)

    p1bedtime = p1Sleeps['sleep']
    ax1 = fig.add_subplot(2, 3, 1)
    n, bins, patches = pylab.hist(p1bedtime, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p1bedtime).mean()
    sigma = np.array(p1bedtime).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax1.plot(bins, y, 'k--', linewidth=3.5)
    ax1.set_ylabel("Participant 3", fontsize=16)
    ax1.set_xlabel("Bedtime", fontsize=16)
    ax1.set_xticks([])
    ax1.set_yticks([])

    p1waketime = p1Sleeps['wakeup']
    ax2 = fig.add_subplot(2, 3, 2)
    n, bins, patches = pylab.hist(p1waketime, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p1waketime).mean()
    sigma = np.array(p1waketime).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax2.plot(bins, y, 'k--', linewidth=3.5)
    ax2.set_xlabel("Waketime", fontsize=16)
    ax2.set_xticks([])
    ax2.set_yticks([])

    p1duration = p1Sleeps['duration']
    ax3 = fig.add_subplot(2, 3, 3)
    n, bins, patches = pylab.hist(p1duration, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p1duration).mean()
    sigma = np.array(p1duration).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax3.plot(bins, y, 'k--', linewidth=3.5)
    ax3.set_xlabel("Duration", fontsize=16)
    ax3.set_xticks([])
    ax3.set_yticks([])

    p2bedtime = p2Sleeps['sleep']
    ax4 = fig.add_subplot(2, 3, 4)
    n, bins, patches = pylab.hist(p2bedtime, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p2bedtime).mean()
    sigma = np.array(p2bedtime).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax4.plot(bins, y, 'k--', linewidth=3.5)
    ax4.set_ylabel("Participant 12", fontsize=16)
    ax4.set_xlabel("Bedtime", fontsize=16)
    ax4.set_xticks([])
    ax4.set_yticks([])

    p2waketime = p2Sleeps['wakeup']
    ax5 = fig.add_subplot(2, 3, 5)
    n, bins, patches = pylab.hist(p2waketime, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p2waketime).mean()
    sigma = np.array(p2waketime).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax5.plot(bins, y, 'k--', linewidth=3.5)
    ax5.set_xlabel("Waketime", fontsize=16)
    ax5.set_xticks([])
    ax5.set_yticks([])

    p2duration = p2Sleeps['duration']
    ax6 = fig.add_subplot(2, 3, 6)
    n, bins, patches = pylab.hist(p2duration, bins=20, normed=1)
    plt.setp(patches, 'facecolor', 'g', 'alpha', 0.85)
    mu = np.array(p2duration).mean()
    sigma = np.array(p2duration).std()
    y = pylab.normpdf(bins, mu, sigma)
    ax6.plot(bins, y, 'k--', linewidth=3.5)
    ax6.set_xlabel("Duration", fontsize=16)
    ax6.set_xticks([])
    ax6.set_yticks([])

    plt.show()


def getGeneralModelAccuracyFig():
    """
    model = GeneralModel(0.1, True)
    accList = list()
    for dbFile in model.dbList:
        combiner = DataCombiner(dbFile)
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0)
        fullData, fullLabel = generator.generateFullDataset(t=12)
        predicts = model.clf.predict(fullData)
        accList.append(accuracy_score(fullLabel, predicts))
    with open("values/generalModelAccuracy.pickle", 'w') as f:
        pickle.dump(accList, f)
    """

    with open("values/generalModelAccuracy.pickle", 'r') as f:
        accList = pickle.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colorList = ['royalblue', 'darkkhaki']
    margin = 0.2
    bar_width = 1.0 - margin
    index = np.arange(len(accList))
    ax.bar(index, accList, bar_width, color = colorList[0])
    #plt.legend((bar1[0], bar2[0]), ('Bedtime', 'Waketime'), 0)
    ax.set_xlabel('Participant ID', fontsize=16)
    ax.set_ylabel('Prediction Accuracy', fontsize=16)
    plt.xlim(0, len(accList))
    plt.ylim(0.8, 1.)
    plt.xticks(np.arange(len(accList)) + bar_width/2, range(1, len(accList)+1))
    #plt.legend(loc='upper right', fontsize='x-large')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    #getLeaveOneDayOutClassificationCdfFig()
    #getSleepErrorFig("values/sleepErrors.pickle")
    #getSleepErrorFig("values/leaveOneDayOutSleepErrors.pickle")
    #getNUSError()
    #getSleepErrorFig("values/nusSleepErrors.pickle")
    #getCompareDurationOfNusFig("values/compareNusDuration.pickle")
    #getSleepStatisticFig()
    #getScreenOnStatisticFig()

    #getPredictionErrorVariance()
    #getPredictionErrorVarianceFig('sleep')
    #getScreenOnTimeAndLoggedVarianceFig('duration')

    #getDistanceSleepVarianceFig("wakeup")
    #getIrregularSleepFig()
    #getIrregularWakeupFig()
    #getIrregularDurationFig()

    #plotSleepHistogramFig()

    getGeneralModelAccuracyFig()

