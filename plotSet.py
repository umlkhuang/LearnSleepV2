import numpy as np
import matplotlib.pyplot as plt

from glob import glob 
from dataCombiner import DataCombiner 
from dataGenerator import DataGenerator 
from sleepClassifiers import SleepClassifiers 
from dataSegment import DataSegment 
from matplotlib.patches import Rectangle 
from datetime import datetime, timedelta 
from matplotlib.dates import date2num
from time import time
from sleepClusters import * 
from sklearn import metrics 
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib.transforms import BlendedGenericTransform 
from sklearn.ensemble import RandomForestClassifier


def plotAllUserAccuracyAndF1(inputs): 
    userNum = len(inputs) 
    if userNum == 0:
        return 
    algorithmNum = len(inputs[0]) 
    algorithmList = inputs[0].keys() 
    # Plot for each algorithm 
    for algorithmIdx in range(algorithmNum): 
        means_acc = ()
        std_acc = () 
        means_f1 = () 
        std_f1 = () 
        algorithmName = algorithmList[algorithmIdx] 
        for userData in inputs:
            valueList = userData[algorithmName] 
            meanValueDict = valueList[0] 
            stdValueDict = valueList[1] 
            means_acc += (meanValueDict['accuracy'], )
            std_acc += (stdValueDict['accuracy'], )
            means_f1 += (meanValueDict['f1'], )
            std_f1 += (stdValueDict['f1'], ) 
        fig, ax = plt.subplots() 
        index = np.arange(userNum) 
        bar_width = 0.25 
        error_config = {'ecolor': 'black', 'elinewidth': 1}  
        
        rects1 = plt.bar(index, means_acc, bar_width, color = 'b',
                         yerr = std_acc, error_kw = error_config, label = 'Accuracy') 
        
        rects2 = plt.bar(index + bar_width, means_f1, bar_width, color = 'r', 
                         yerr = std_f1, error_kw = error_config, label = 'F1') 
        
        plt.xlabel('User ID') 
        plt.ylabel('Values') 
        plt.title(algorithmName + " Classification Results") 
        plt.xlim(0, userNum)
        plt.xticks(index + bar_width, range(1, userNum + 1)) 
        plt.ylim(0.65, 1.1) 
        plt.yticks(np.arange(0.65, 1.1, 0.05))
        plt.legend(loc = 2) 
        plt.tight_layout() 
        plt.show() 

def plotCompareDifferentAlgorithms(inputs):
    userNum = len(inputs) 
    if userNum == 0:
        return 
    algorithmNum = len(inputs[0]) 
    algorithmList = inputs[0].keys() 
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    plt.hold(True) 
    colorList = ['darkkhaki','royalblue', 'r', 'm', 'b', 'y', 'c', 'g'] 
    margin = 0.25
    bar_width = (1.0 - margin) / algorithmNum 
    index = np.arange(userNum) 
    error_config = {'ecolor': 'black', 'elinewidth': 1.5}  
    for algorithmIdx in range(algorithmNum): 
        means_acc = ()
        std_acc = () 
        algorithmName = algorithmList[algorithmIdx] 
        for userData in inputs:
            valueList = userData[algorithmName]
            meanValueDict = valueList[0]
            stdValueDict = valueList[1]
            means_acc += (meanValueDict['accuracy'], )
            std_acc += (stdValueDict['accuracy'], ) 
        plt.bar(index + bar_width * algorithmIdx, means_acc, bar_width, color = colorList[algorithmIdx],
                yerr = std_acc, error_kw = error_config, label = algorithmName) 
    plt.hold(False)
    plt.xlabel('User ID') 
    plt.ylabel('Average Classification Accuracy') 
    plt.xlim(0, userNum)
    plt.xticks(index + bar_width * algorithmNum / 2.0, range(1, userNum + 1)) 
    plt.ylim(0.70, 1.1) 
    plt.yticks(np.arange(0.70, 1.1, 0.05))
    plt.tight_layout() 
    plt.legend(loc = 2)
    plt.show()

def plotSleepDuration(data, left, right, smoothStart):
    num = len(data) 
    fig, axarr = plt.subplots(num, 1, sharex=True, figsize = (15.0, 2.0)) 
    leftMargin = left
    rightMargin = right 
    for idx in range(num):
        sleeps = data[idx] 
        for sleep in sleeps: 
            if sleep['end'] == -1: 
                sleep['end'] = rightMargin 
                sleep['duration'] = (rightMargin - sleep['start']).total_seconds() / 60 
            startX = int((sleep['start'] - leftMargin).total_seconds()) 
            endX = int((sleep['end'] - leftMargin).total_seconds()) 
            axarr[idx].add_patch(Rectangle((startX, 0), int(sleep['duration']*60), 0.01, facecolor='green'))
            axarr[idx].set_xlim(0, int((rightMargin - leftMargin).total_seconds())) 
            axarr[idx].get_xaxis().set_ticks([])
            axarr[idx].set_ylim(0, 0.01)
            axarr[idx].get_yaxis().set_ticks([]) 
            if idx == 0:
                axarr[idx].set_ylabel('Sleep Log', fontsize = 14)
            elif idx == 1:
                axarr[idx].set_ylabel('Sensing Truth', fontsize = 14)
            else:
                axarr[idx].set_ylabel('Smooth ' + str(idx - 2 + smoothStart), fontsize = 14)
            axarr[idx].yaxis.get_label().set_rotation('horizontal') 
    plt.show() 


def getFeatureCombinationTable():
    dbList = glob('./data/*.db') 
    totalRet = []
    for c in range(1, 16): 
        t1 = time()
        results = [] 
        for dbName in dbList: 
            #if dbName == './data/7bf325b206ffab8a1fdf924d1901951.db':
            #    continue 
            combiner = DataCombiner(dbName) 
            combiner.combineData()
            generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3)
            trainData, trainLabel = generator.generateTrainingDataSet(t = c)
            testData, testLabel = generator.generateTestDataSet(t = c) 
            fullData, fullLabel = generator.generateFullDataset(t = c) 
            classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel)
            svmCrossValidationResults = classifier.SVMClassifier(crossValidation = True) 
            results.append(round(svmCrossValidationResults[0]['accuracy'], 3)) 
            #dtCrossValidationResults = classifier.DTClassifier(crossValidation = True)
            #results.append(round(dtCrossValidationResults[0]['accuracy'], 3))
            del(combiner)
            del(generator) 
            del(classifier) 
        totalRet.append(results)
        t2 = time()
        print "\n\n================  C = %d. Total process time = %s\n" % (c, str(t2 - t1)) 
    for idx in range(1, len(totalRet)+1):
        row = totalRet[idx-1]
        print "\\textbf{C%d}" % idx, 
        for data in row:
            print "& %4.3f" % data, 
        print "\\\\ " 
             
def getCompareAlgorithmsFig():
    dbList = glob('./data/*.db') 
    totalResult = [] 
    for dbName in dbList: 
        #if dbName == './data/a02bc6709a9e4f2ea74029d9636d3da7.db':
        #    continue 
        #if dbName == './data/7bf325b206ffab8a1fdf924d1901951.db':
        #    continue
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3) 
        trainData, trainLabel = generator.generateTrainingDataSet(t = 12)
        testData, testLabel = generator.generateTestDataSet(t = 12) 
        fullData, fullLabel = generator.generateFullDataset(t = 12) 
        classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel) 
        resultDict = dict() 
        svmCrossValidationResults = classifier.SVMClassifier(crossValidation = True) 
        resultDict['SVM'] = svmCrossValidationResults 
        dtCrossValidationResults = classifier.DTClassifier(crossValidation = True) 
        resultDict['Decision Tree'] = dtCrossValidationResults 
        lrCrossValidationResults = classifier.LRClassifier(crossValidation = True) 
        resultDict['Logistic Regression'] = lrCrossValidationResults 
        totalResult.append(resultDict) 
        del(combiner)
        del(generator)
        del(classifier) 
    plotCompareDifferentAlgorithms(totalResult) 
    
def getDatasetStatisticTable():
    days = [] 
    dataNo = [] 
    sleepNo = [] 
    dbList = glob('./data/*.db') 
    for dbName in dbList:
        print "===================  " + dbName
        totalSleep = 0 
        combiner = DataCombiner(dbName) 
        combiner.combineData() 
        days.append(len(combiner.sleepData))
        dataNo.append(len(combiner.combinedDataList)) 
        for instance in combiner.combinedDataList:
            totalSleep += instance.getSleepStatus() 
        sleepNo.append(totalSleep) 
        del(combiner) 
    print "\n\n\n"
    print "\\textbf{Participated Days}",
    for day in days:
        print "& %d" % day, 
    print "\\\\ \\hline" 
    print "\\textbf{Total Sensing Data}", 
    for num in dataNo:
        print "& %d" % num,
    print "\\\\ \hline" 
    print "\\textbf{Sleep Sensing Data}",
    for sleep in sleepNo:
        print "& %d" % sleep,
    print "\\\\ \hline" 
    print "\n\n"
    print "Total days = %d" % sum(days)
    print "Total sensing data = %d" % sum(dataNo) 
    

def getChiSquareTestFig():
    dbList = glob('./data/*.db') 
    #dbList = ['./data/ecdbeb2bc610d72e726b58a04e463a3f.db']
    fig = plt.figure() 
    ax = fig.add_subplot(111) 
    plt.hold(True)
    for dbName in dbList:
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3) 
        trainData, trainLabel = generator.generateTrainingDataSet(t = 12)
        testData, testLabel = generator.generateTestDataSet(t = 12) 
        fullData, fullLabel = generator.generateFullDataset(t = 12)
        classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel)
        chi_val, p_val = classifier.chi2Test() 
        whereAreNaNs = np.isnan(p_val)
        for idx in range(len(p_val)):
            if p_val[idx] == 0.0:
                p_val[idx] = 1.0e-250 
        p_val[whereAreNaNs] = 1.0e-250 
        scores = -np.log10(p_val) 
        #scores /= scores.max()
        plt.plot(np.arange(1, 11), scores, 'k-', linewidth = 1.5) 
    #plt.axhline(-np.log10(0.05))
    plt.hold(False)
    ax.set_xlabel('Context Feature Id')
    ax.set_ylabel('Univariate score ($-Log(p_{value})$)') 
    plt.grid(True)
    plt.show()

def getSmoothResultsFig():
    #dbList = glob('./data/*.db') 
    dbList = ["./data/658ac828bdadbddaa909315ad80ac8.for_smooth"] 
    for dbName in dbList:
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.2) 
        trainData, trainLabel = generator.generateTrainingDataSet(t = 15)
        testData, testLabel = generator.generateTestDataSet(t = 15) 
        fullData, fullLabel = generator.generateFullDataset(t = 15)
        classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel) 
        tmp = list()
        d0 = combiner.getSleepLogFromTS(generator.testCreateTimeList[0]) 
        predict, score = classifier.DTClassifier() 
        d1 = classifier.getSleepTimeAndDuration(testLabel, generator.testCreateTimeList) 
        tmp.append(d0)
        tmp.append(d1)
        for num in range(0, 5):
            smoothedPredicts = classifier.smoothPrediction(predict, num) 
            classifier.showClassificationInfo(smoothedPredicts) 
            ret = classifier.getSleepTimeAndDuration(smoothedPredicts, generator.testCreateTimeList)
            tmp.append(ret) 
        plotSleepDuration(tmp, generator.testCreateTimeList[0], generator.testCreateTimeList[-1], 0) 

def dt2num(dt):
    hour = dt.hour
    minute = dt.minute
    if hour < 12:
        return (24 + hour) * 60 + minute 
    else:
        return hour * 60 + minute 

def getSleepTimeDurationEvaluationFig():
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"
    dbName = "./data/6e44881f5af5d54a452b99f57899a7.db" 
    combiner = DataCombiner(dbName) 
    combiner.combineData()
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3) 
    trainData, trainLabel = generator.generateTrainingDataSet(t = 15)
    testData, testLabel = generator.generateTestDataSet(t = 15) 
    fullData, fullLabel = generator.generateFullDataset(t = 15)
    classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel) 
    d0 = combiner.getSleepLogFromTS(generator.testCreateTimeList[0]) 
    for item in d0:
        print item
    predict, score = classifier.DTClassifier() 
    smoothedPredicts = classifier.smoothPrediction(predict, 4) 
    d1 = classifier.getSleepTimeAndDuration(smoothedPredicts, generator.testCreateTimeList) 
    
    plotSleepDuration([d0, d1], generator.testCreateTimeList[0], generator.testCreateTimeList[-1], 0) 
    
    days = np.arange(1, len(d0)+1) 
    fig = plt.figure()
    par2 = fig.add_subplot(111)
    ax = par2.twinx() 
    bar_width = 0.35 
    plt.hold(True)
    trueSleepTime = []
    trueSleepDuration = []
    predictSleepTime = []
    predictSleepDuration = [] 
    durationDelta = [] 
    timeDelta = [] 
    for item in d0:
        trueSleepTime.append(dt2num(item['start'].time()))
        trueSleepDuration.append(item['duration']) 
    for item in d1:
        predictSleepTime.append(dt2num(item['start'].time()))
        predictSleepDuration.append(item['duration']) 
    for idx in range(len(d0)):
        durationDelta.append(abs(d0[idx]['duration'] - d1[idx]['duration'])) 
        timeDelta.append(abs((d0[idx]['start'] - d1[idx]['start']).total_seconds()) / 60) 
    durationDelta = np.array(durationDelta)
    timeDelta = np.array(timeDelta) 
    print "Duration: min = %f, max = %f, mean = %f, std = %f" % (durationDelta.min(), durationDelta.max(), durationDelta.mean(), durationDelta.std()) 
    print "Time: minmin = %f, max = %f, mean = %f, std = %f" % (timeDelta.min(), timeDelta.max(), timeDelta.mean(), timeDelta.std()) 
    p3 = par2.bar(days, trueSleepDuration, bar_width, color = 'darkkhaki', label = 'Logged Sleep Duration')
    p4 = par2.bar(days+bar_width, predictSleepDuration, bar_width, color = 'royalblue', label = 'Predicted Sleep Duration')
    p1 = ax.plot(days+bar_width, trueSleepTime, 'b--s', linewidth = 1.5, label = 'Logged Bedtime')
    p2 = ax.plot(days+bar_width, predictSleepTime, 'b-s', linewidth = 1.5, label = 'Predicted Bedtime') 
    ax.set_xlim([1, len(d0)+1]) 
    
    # Regular user: ./data/6e44881f5af5d54a452b99f57899a7.db
    # Duration: min = 0.516672, max = 53.349992, mean = 12.466666, std = 13.657169
    # Time: minmin = 1.483336, max = 6.466667, mean = 4.080557, std = 1.581953
    
    ax.set_ylim([1400, 1580]) 
    ax.get_yaxis().set_ticklabels(['23:20', '23:40', '00:00', '00:20', '00:40', '01:00', '01:20', '01:40', '02:00', '02:20'])
    
    
    # Irregular user: ./data/ecdbeb2bc610d72e726b58a04e463a3f.db 
    # Duration: min = 28.683321, max = 212.916654, mean = 127.995825, std = 80.047244 
    # Time: minmin = 0.566676, max = 64.683340, mean = 31.566669, std = 26.179067 
    """
    ax.set_ylim([1250, 1550])
    ax.get_yaxis().set_ticklabels(['20:50', '21:40', '22:30', '23:20', '00:10', '01:00', '01:50'])
    """
    
    plt.xticks(days+bar_width, np.arange(1, len(d0)+1))
    plt.grid(True)
    par2.set_xlabel('Day')
    ax.set_ylabel('Bedtime') 
    par2.set_ylim([0, 750])
    par2.set_ylabel('Sleep Duration (in minutes)') 
    lines = [p3, p4] 
    bars = [p1[0], p2[0]]
    ax.legend(lines, [l.get_label() for l in lines], loc = 2)
    par2.legend(bars, [b.get_label() for b in bars], loc = 1)
    plt.show() 

def getSinglePointClustringMetricTable():
    dbList = glob('./data/*.db') 
    ret = list() 
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        tmp = dict() 
        combiner = DataCombiner(dbName) 
        combiner.combineData() 
        clf = SleepClusters(combiner, 12) 
        predicts = clf.bench_KMeans('k-means++') 
        ARI = metrics.adjusted_rand_score(clf.label, predicts)
        NMI = metrics.normalized_mutual_info_score(clf.label, predicts) 
        newPredicts = clf.adjustPredicts(predicts) 
        acc, f1 = clf.showClassificationInfo(newPredicts) 
        tmp['ARI'] = ARI 
        tmp['NMI'] = NMI 
        tmp['acc'] = acc 
        tmp['f1'] = f1
        ret.append(tmp)
        del(combiner)
        del(clf) 
        del(predicts)
        del(tmp)
    print "\n\n" 
    for idx in range(len(ret)):
        print "%d & %.2f & %.2f & %.2f & %.2f & " % (idx+1, ret[idx]['ARI'], ret[idx]['NMI'], ret[idx]['acc'], ret[idx]['f1'])
    
def getContinueClusteringMetricTable():
    dbList = glob('./data/*.db') 
    ret = list() 
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        tmp = dict() 
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        segmenter = DataSegment(combiner.combinedDataList, 0.3)
        estimator = KMeans(n_clusters = 2, init = 'k-means++', n_jobs = -1, verbose = 0)
        data = scale(segmenter.data)
        predicts = estimator.fit(data).labels_ 
        ARI = metrics.adjusted_rand_score(segmenter.label, predicts)
        NMI = metrics.normalized_mutual_info_score(segmenter.label, predicts) 
        newPredicts = segmenter.newAdjustPredicts(predicts) 
        acc, f1 = segmenter.showClassificationInfo(newPredicts) 
        tmp['ARI'] = ARI 
        tmp['NMI'] = NMI 
        tmp['acc'] = acc 
        tmp['f1'] = f1
        ret.append(tmp)
        del(combiner)
        del(segmenter) 
        del(estimator)
        del(tmp)
    print "\n\n" 
    for idx in range(len(ret)):
        print "%.2f & %.2f & %.2f & %.2f \\\\" % (ret[idx]['ARI'], ret[idx]['NMI'], ret[idx]['acc'], ret[idx]['f1'])

def clusterU1SleepTimeDurationFig():
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"
    dbName = "./data/6e44881f5af5d54a452b99f57899a7.db"
    combiner = DataCombiner(dbName) 
    combiner.combineData()
    segmenter = DataSegment(combiner.combinedDataList, 0.3) 
    estimator = KMeans(n_clusters = 2, init = 'k-means++', n_jobs = -1, verbose = 0)
    data = scale(segmenter.data) 
    predicts = estimator.fit(data).labels_  
    newPredicts = segmenter.newAdjustPredicts(predicts) 
    d0 = combiner.getSleepLogFromTS(segmenter.timeList[segmenter.continueDataBreakId]['start']) 
    for item in d0:
        print item
    print "\n"
    d1 = segmenter.getSleepTimeAndDuration(newPredicts) 
    for item in d1:
        print item
    print "\n"
    
    #dummy = dict() 
    #dummy['start'] = d1[2]['start']
    #dummy['end'] = 0
    #dummy['duration'] = 0 
    #d0.insert(2, dummy)
    
    days = np.arange(1, len(d0)+1) 
    fig = plt.figure()
    par2 = fig.add_subplot(111)
    ax = par2.twinx() 
    bar_width = 0.35 
    plt.hold(True)
    trueSleepTime = []
    trueSleepDuration = []
    predictSleepTime = []
    predictSleepDuration = [] 
    durationDelta = [] 
    timeDelta = [] 
    for item in d0:
        trueSleepTime.append(dt2num(item['start'].time()))
        trueSleepDuration.append(item['duration']) 
    for item in d1:
        predictSleepTime.append(dt2num(item['start'].time()))
        predictSleepDuration.append(item['duration']) 
    for idx in range(len(d0)): 
        if idx == 2:
            continue 
        durationDelta.append(abs(d0[idx]['duration'] - d1[idx]['duration'])) 
        timeDelta.append(abs((d0[idx]['start'] - d1[idx]['start']).total_seconds()) / 60) 
    durationDelta = np.array(durationDelta)
    timeDelta = np.array(timeDelta) 
    print "Duration: min = %f, max = %f, mean = %f, std = %f" % (durationDelta.min(), durationDelta.max(), durationDelta.mean(), durationDelta.std()) 
    print "Time: minmin = %f, max = %f, mean = %f, std = %f" % (timeDelta.min(), timeDelta.max(), timeDelta.mean(), timeDelta.std()) 
    p3 = par2.bar(days, trueSleepDuration, bar_width, color = 'darkkhaki', label = 'Logged Sleep Duration')
    p4 = par2.bar(days+bar_width, predictSleepDuration, bar_width, color = 'royalblue', label = 'Predicted Sleep Duration')
    p1 = ax.plot(days+bar_width, trueSleepTime, 'b--s', linewidth = 1.5, label = 'Logged Bedtime')
    p2 = ax.plot(days+bar_width, predictSleepTime, 'b-s', linewidth = 1.5, label = 'Predicted Bedtime') 
    ax.set_xlim([1, len(d0)+1]) 
    ax.set_ylim([1380, 1560]) 
    ax.get_yaxis().set_ticklabels(['23:00', '23:20', '23:40', '00:00', '00:20', '00:40', '01:00', '01:20', '01:40', '02:00'])
    plt.xticks(days+bar_width, np.arange(1, len(d0)+1))
    plt.grid(True)
    par2.set_xlabel('Day')
    ax.set_ylabel('Bedtime') 
    par2.set_ylim([0, 750])
    par2.set_ylabel('Sleep Duration (in minutes)') 
    lines = [p3, p4] 
    bars = [p1[0], p2[0]]
    ax.legend(lines, [l.get_label() for l in lines], loc = 2)
    par2.legend(bars, [b.get_label() for b in bars], loc = 1)
    plt.show()

def clusterU2SleepTimeDurationFig():
    dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"
    combiner = DataCombiner(dbName) 
    combiner.combineData()
    segmenter = DataSegment(combiner.combinedDataList, 0.165) 
    estimator = KMeans(n_clusters = 2, init = 'k-means++', n_jobs = -1, verbose = 0)
    data = scale(segmenter.data) 
    predicts = estimator.fit(data).labels_  
    newPredicts = segmenter.newAdjustPredicts(predicts) 
    d0 = combiner.getSleepLogFromTS(segmenter.timeList[segmenter.continueDataBreakId]['start']) 
    #d0 = segmenter.getSleepTimeAndDuration(segmenter.label) 
    for item in d0:
        print item
    print "\n"
    d1 = segmenter.getSleepTimeAndDuration(newPredicts) 
    for item in d1:
        print item
    print "\n"
    
    days = np.arange(1, len(d0)+1) 
    fig = plt.figure()
    par2 = fig.add_subplot(111)
    ax = par2.twinx() 
    bar_width = 0.35 
    plt.hold(True)
    trueSleepTime = []
    trueSleepDuration = []
    predictSleepTime = []
    predictSleepDuration = [] 
    durationDelta = [] 
    timeDelta = [] 
    for item in d0:
        trueSleepTime.append(dt2num(item['start'].time()))
        trueSleepDuration.append(item['duration']) 
    for item in d1:
        predictSleepTime.append(dt2num(item['start'].time()))
        predictSleepDuration.append(item['duration']) 
    for idx in range(len(d0)):
        durationDelta.append(abs(d0[idx]['duration'] - d1[idx]['duration'])) 
        timeDelta.append(abs((d0[idx]['start'] - d1[idx]['start']).total_seconds()) / 60) 
    durationDelta = np.array(durationDelta)
    timeDelta = np.array(timeDelta) 
    print "Duration: min = %f, max = %f, mean = %f, std = %f" % (durationDelta.min(), durationDelta.max(), durationDelta.mean(), durationDelta.std()) 
    print "Time: minmin = %f, max = %f, mean = %f, std = %f" % (timeDelta.min(), timeDelta.max(), timeDelta.mean(), timeDelta.std()) 
    p3 = par2.bar(days, trueSleepDuration, bar_width, color = 'darkkhaki', label = 'Logged Sleep Duration')
    p4 = par2.bar(days+bar_width, predictSleepDuration, bar_width, color = 'royalblue', label = 'Predicted Sleep Duration')
    p1 = ax.plot(days+bar_width, trueSleepTime, 'b--s', linewidth = 1.5, label = 'Logged Sleep Time')
    p2 = ax.plot(days+bar_width, predictSleepTime, 'b-s', linewidth = 1.5, label = 'Predicted Sleep Time') 
    ax.set_xlim([1, len(d0)+1]) 
    ax.set_ylim([1250, 1550])
    ax.get_yaxis().set_ticklabels(['20:50', '21:40', '22:30', '23:20', '00:10', '01:00', '01:50'])
    plt.xticks(days+bar_width, np.arange(1, len(d0)+1))
    plt.grid(True)
    par2.set_xlabel('Day')
    ax.set_ylabel('Sleep Time') 
    par2.set_ylim([0, 750])
    par2.set_ylabel('Sleep Duration (in minutes)') 
    lines = [p3, p4] 
    bars = [p1[0], p2[0]]
    ax.legend(lines, [l.get_label() for l in lines], loc = 2)
    par2.legend(bars, [b.get_label() for b in bars], loc = 1)
    plt.show()
    
def getClusterCdfFig():
    dbList = glob('./data/*.db') 
    #dbList = ['./data/78a4e254e6fa406ebe1bd3fab8a57499.db'] 
    sleepTime = list() 
    duration = list() 
    for idx in range(len(dbList)):
        dbName = dbList[idx]
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        segmenter = DataSegment(combiner.combinedDataList, 0.0) 
        estimator = KMeans(n_clusters = 2, init = 'k-means++', n_jobs = -1, verbose = 0)
        data = scale(segmenter.data) 
        predicts = estimator.fit(data).labels_  
        newPredicts = segmenter.newAdjustPredicts(predicts) 
        d0Raw = combiner.getSleepLogFromTS(segmenter.timeList[segmenter.continueDataBreakId]['start'])
        #d0Raw = segmenter.getSleepTimeAndDuration(segmenter.label)
        d0 = list()
        for idx in range(len(d0Raw)): 
            item = d0Raw[idx] 
            if item['duration'] >= 120 and item['duration'] <= 750:
                d0.append(item) 
        
        print "=========  d0 ============="
        for item in d0:
            print item
        print "\n"
        
        
        d1 = segmenter.getSleepTimeAndDuration(newPredicts) 
        
        
        print "=========  d1 ============="
        for item in d1:
            print item
        print "Truth: %d , predict: %d" % (len(d0), len(d1))
        print "\n"
        
        
        del(combiner)
        del(segmenter)
        del(estimator)
        
        truthId = 0
        predictId = 0
        while True: 
            if predictId >= len(d1) or truthId >= len(d0):
                break 
            trueSleepTime = d0[truthId]['start'] 
            predSleepTime = d1[predictId]['start'] 
            sleepDelta = abs((predSleepTime - trueSleepTime).total_seconds()) / 60
            if sleepDelta < 200:  
                sleepTime.append(sleepDelta) 
                durationDelta = abs(d0[truthId]['duration'] - d1[predictId]['duration']) 
                if durationDelta > 120:
                    print "TT: " + str(d0[truthId]['start']) + ", PT: " + str(d1[predictId]['start']) + ", TD: " + str(d0[truthId]['duration']) + ", PD: " + str(d1[predictId]['duration'])
                duration.append(durationDelta) 
                truthId += 1
                predictId += 1 
            elif d0[truthId]['start'] < d1[predictId]['start']:
                truthId += 1 
            else:
                predictId += 1
    sleepTime = sorted(sleepTime)
    duration = sorted(duration) 
    print "Mean Sleep time error: %f" % (sum(sleepTime) / len(sleepTime))
    print "Mean Sleep duration error: %f" % (sum(duration) / len(duration)) 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True) 
    maxSleepDelta = int(sleepTime[-1])
    dx = 1 
    X = np.arange(0, maxSleepDelta + 1, 1) 
    Y = list() 
    sleepLen = len(sleepTime)
    for value in X:
        cdf = sum(1 for i in sleepTime if i <= value) * 1.0 / sleepLen 
        Y.append(cdf) 
    plt.plot(X, Y, 'o') 
    ax.set_xticks(np.arange(0, maxSleepDelta + 10, 20))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax.set_xlabel('Bedtime error (in minutes)')
    ax.set_ylabel('CDF')
    plt.show() 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True) 
    maxDurationDelta = int(duration[-1])
    dx = 1 
    X = np.arange(0, maxDurationDelta + 1, 1) 
    Y = list() 
    durationLen = len(duration)
    for value in X:
        cdf = sum(1 for i in duration if i <= value) * 1.0 / durationLen 
        Y.append(cdf) 
    plt.plot(X, Y, 'o') 
    ax.set_xticks(np.arange(0, maxDurationDelta + 10, 30))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax.set_xlabel('Sleep duration error (in minutes)')
    ax.set_ylabel('CDF')
    print "maxDurationDelta = " + str(maxDurationDelta) 
    plt.show() 

def processSleeps(d1Raw, minDuration = 120, combinThreshold = 30):
    d1 = list()
    for idx in range(len(d1Raw)):
        item = d1Raw[idx]
        #print item
        if item['duration'] <= 750:
            if len(d1) >= 1:
                if (item['start'] - d1[-1]['end']).total_seconds() <= combinThreshold * 60:
                    d1[-1]['end'] = item['end']
                    d1[-1]['duration'] = (d1[-1]['end'] - d1[-1]['start']).total_seconds() / 60
                else:
                    d1.append(item)
            else:
                d1.append(item)
            """
            if item['duration'] > minDuration:
                d1.append(item)
            elif len(d1) >= 1 and (item['start'] - d1[-1]['end']).total_seconds() <= combinThreshold * 60:
                d1[-1]['end'] = item['end']
                d1[-1]['duration'] = (d1[-1]['end'] - d1[-1]['start']).total_seconds() / 60
                #print str(item['start']) + "==========   combined with previous"
            elif idx < (len(d1Raw) - 1) and (d1Raw[idx + 1]['start'] - item['end']).total_seconds() <= combinThreshold * 60:
                d1Raw[idx + 1]['start'] = item['start']
                d1Raw[idx + 1]['duration'] = (d1Raw[idx + 1]['end'] - d1Raw[idx + 1]['start']).total_seconds() / 60
                #print str(item['end']) + "==========   combined with next one"
            """
    for idx in range(len(d1) - 1, -1, -1):
        if d1[idx]['duration'] < minDuration:
            d1.pop(idx)
    return d1

def getErrors(d0, d1, sleepTime, wakeTime, duration):
    truthId = 0
    predictId = 0
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
            if durationDelta > 75 or sleepDelta > 30:
                print "BedtimeDelta: " + str(sleepDelta) + " DurDelta: " + str(durationDelta) + " TT: " + str(d0[truthId]['start']) + ", PT: " + str(d1[predictId]['start']) + ", TD: " + str(d0[truthId]['duration']) + ", PD: " + str(d1[predictId]['duration'])
            duration.append(durationDelta)
            truthId += 1
            predictId += 1
        elif d0[truthId]['start'] < d1[predictId]['start']:
            truthId += 1
        else:
            predictId += 1
    print ""

def getClassificationCdfFig():
    dbList = glob('./data/*.db')
    #dbList = ['./data/c6bd3bfcbfdd5f17fb9d23484b8ab95.db']
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
        kf = cross_validation.KFold(len(data), n_folds = 10)
        d0Raw = combiner.getSleepLogFromTS(generator.fullCreateTimeList[0]) 
        #d0Raw = classifier.getSleepTimeAndDuration(label, generator.fullCreateTimeList) 
        d0 = list()
        for idx in range(len(d0Raw)): 
            item = d0Raw[idx] 
            if item['duration'] >= 120 and item['duration'] <= 750:
                d0.append(item) 
        print "===========  d0  ==========="
        for item in d0:
            print item
        print "\n"
        
        predicts = []  
        for train_idx, test_idx in kf:
            X_train, X_test = data[train_idx], data[test_idx]
            Y_train, Y_test = label[train_idx], label[test_idx] 
            #clf = DecisionTreeClassifier(criterion = 'entropy')
            clf = RandomForestClassifier(n_estimators=35, criterion='entropy', n_jobs=-1)
            clf.fit(X_train, Y_train) 
            predicts += list(clf.predict(X_test))
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
    print "Min Sleep time error: %f" % (np.min(sleepTime))
    print "Max Sleep time error: %f" % (np.max(sleepTime))
    print "Mean Sleep time error: %f" % (sum(sleepTime) / len(sleepTime))
    print "STD Sleep time error: %f" % (np.std(sleepTime))
    print ""
    print "Min Wakeup time error: %f" % (np.min(wakeTime))
    print "Max Wakeup time error: %f" % (np.max(wakeTime))
    print "Mean Wakeup time error: %f" % (sum(wakeTime) / len(wakeTime))
    print "STD Wakeup time error: %f" % (np.std(wakeTime))
    print ""
    print "Min Sleep duration error: %f" % (np.min(duration))
    print "Max Sleep duration error: %f" % (np.max(duration))
    print "Mean Sleep duration error: %f" % (sum(duration) / len(duration))
    print "STD Sleep duration error: %f" % (np.std(duration))

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True) 
    maxSleepDelta = int(sleepTime[-1])
    X = np.arange(0, maxSleepDelta + 1, 1) 
    Y = list() 
    sleepLen = len(sleepTime)
    for value in X:
        cdf = sum(1 for i in sleepTime if i <= value) * 1.0 / sleepLen 
        Y.append(cdf) 
    plt.plot(X, Y, 'o') 
    #ax.set_xticks(np.arange(0, maxSleepDelta + 10, 20))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax.set_xlabel('Bedtime error (in minutes)')
    ax.set_ylabel('CDF')
    plt.show() 
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True) 
    #maxDurationDelta = 330 
    maxDurationDelta = int(duration[-1])
    X = np.arange(0, maxDurationDelta + 1, 1) 
    Y = list() 
    durationLen = len(duration)
    for value in X:
        cdf = sum(1 for i in duration if i <= value) * 1.0 / durationLen 
        Y.append(cdf) 
    plt.plot(X, Y, 'x')
    #ax.set_xticks(np.arange(0, maxDurationDelta + 10, 30))
    ax.set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax.set_xlabel('Sleep duration error (in minutes)')
    ax.set_ylabel('CDF')
    plt.show()
    """
        
def plotSleepTrackingProjects():
    fig = plt.figure(1) 
    ax = SubplotZero(fig, 111) 
    fig.add_subplot(ax)
    
    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True) 
    for direction in ["left", "right", "bottom", "top"]:
        ax.axis[direction].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim([-1., 1.])
    ax.set_ylim([-1., 1.])
    
    ax.text(0, 1.07, 'User Cooperation\n(Short term monitoring)', va='bottom', ha='center')
    ax.text(0, -1.17, 'Passive\n(Long term monitoring)', va='bottom', ha='center')
    ax.text(-1.02, 0, 'Less Detail\n(Bedtime, duration)', va='center', ha='right')
    ax.text(1.07, 0, 'More Detail\n(Quality)', va='center', ha='left')
    
    ax.annotate('Polysomnography', xy=(0.9, 0.9),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Zoe', xy=(0.5, 0.8),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Fitbit\nJawbone', xy=(0.23, 0.4),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Basis', xy=(0.43, 0.5),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Sleepbot', xy=(0.35, 0.25),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('iSleep', xy=(0.45, 0.15),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Toss & Turn', xy=(0.2, -0.2),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('SleepCollector', xy=(0.07, -0.7),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc="darkkhaki", ec=(1., .5, .5)))
    
    ax.annotate('Beddit', xy=(0.6, -0.3),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('Lullaby', xy=(0.6, -0.7),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    ax.annotate('BES', xy=(-0.65, -0.83),  xycoords='data',
                xytext=(0, 0), textcoords='offset points',
                size=12, 
                bbox=dict(boxstyle="round", fc=(1.0, 0.7, 0.7), ec=(1., .5, .5)))
    
    plt.show() 
    
    
    


if __name__ == "__main__": 
    getDatasetStatisticTable()
    #getFeatureCombinationTable() 
    #getChiSquareTestFig()
    #getCompareAlgorithmsFig() 
    #getSmoothResultsFig()                      # Do NOT run again!
    #getSleepTimeDurationEvaluationFig()        # Do NOT run again!
    #getSinglePointClustringMetricTable()
    #getContinueClusteringMetricTable()
    #clusterU1SleepTimeDurationFig()            # Do NOT run again!
    #clusterU2SleepTimeDurationFig()            # Do NOT run again! 
    #getClusterCdfFig() 
    getClassificationCdfFig()
    
    
    
    #plotSleepTrackingProjects()
    
    
