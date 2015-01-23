from instance import CombinedData 
from sleepClassifiers import SleepClassifiers 
from database import DatabaseHelper


class DataCombiner(object):
    """
    The class that used to combine all raw data and pre-process the raw data.
    It can add functions to prepare the data for learning. 
    """
    def __init__(self, dbName): 
        self.databaseFileName = dbName
        self.databaseHelper = DatabaseHelper(self.databaseFileName)
        self.sensingData = self.databaseHelper.getSensingData()
        self.sleepData = self.databaseHelper.getSleepLog()
        self.sysData = self.databaseHelper.getSystemEvents() 
        self.combinedDataList = list() 
        self.sleepLogPtr = 0
        self.sysEventLogPtr = 0 
        print "[1] In data combiner! Parsed DB file: " + dbName 
        print "No. of Sleep Logs = " + str(len(self.sleepData)) 
        
    def combineData(self):
        """
        Combine raw data and save them in a list, each record is an instance of
        CombinedData class. 
        """
        sensingDataLen = len(self.sensingData)
        print "Total sensing data records = " + str(sensingDataLen) 
        for idx in range(0, sensingDataLen):
            instance = self.sensingData[idx] 
            #self.roomDetector.parseSensingDataInstanceForRoom(instance) 
            combinedData = CombinedData(instance) 
            isSleep = self.getSleepStatusForSensingInstance(instance) 
            # Filter all records that has no sleep log to match 
            if isSleep < 0:
                continue 
            combinedData.setSleepStatus(isSleep) 
            #print str(combinedData.createTime) + ", IsSleep = " + str(combinedData.isSleep) + ", ScreenOnTime = " + str(combinedData.screenOnSeconds) 
            self.combinedDataList.append(combinedData) 
    
    def getSleepStatusForSensingInstance(self, sensingInstance):
        """
        Compare the time stamp of the sensing data with the sleep log, get if the user
        is sleep when log this sensing data. 
        Return value is boolean. 1 = isSleep, 0 = NotSleep, -1 = missing sleep log  
        """
        createTime = sensingInstance.createTime 
        while self.sleepLogPtr < len(self.sleepData): 
            currentSleepTime = self.sleepData[self.sleepLogPtr].sleepTime 
            currentWakeTime = self.sleepData[self.sleepLogPtr].wakeupTime 
            # The case that the sensing data time stamp is between two sleep logs  
            if createTime < currentSleepTime: 
                if self.sleepLogPtr == 0:
                    delta = currentSleepTime - createTime
                    if delta.total_seconds() <= 4 * 3600:
                        return 0
                    else:
                        return -1
                else:
                    return 0 
            # The case that a user has not log his wake up time yet, still sleep 
            elif currentWakeTime is None: 
                return -1 
            # The case that this sensing log is between the sleep time 
            elif createTime < currentWakeTime: 
                return 1 
            # We need to check the interval between two consecutive sleep logs and 
            # decide whether we need to move the sleep log pointer 
            else: 
                nextSleepLogId = self.sleepLogPtr + 1 
                if nextSleepLogId < len(self.sleepData):
                    nextSleepTime = self.sleepData[nextSleepLogId].sleepTime 
                    # In case that we will stuck when the interval is greater than 24 hours  
                    if createTime > nextSleepTime:
                        self.sleepLogPtr += 1
                        continue 
                    interval = nextSleepTime - currentWakeTime 
                    # The interval is greater than 1 day, there is a missing sleep log 
                    if interval.days >= 1:
                        delta = nextSleepTime - createTime
                        if delta.total_seconds() <= 4 * 3600:
                            return 0
                        else:
                            return -1
                    # Move the sleep log pointer to the next record and continue 
                    else:
                        self.sleepLogPtr += 1
                        continue 
                # No next sleep log, need to filter the data 
                else:
                    if (createTime - currentWakeTime).total_seconds() >= 4 * 3600:
                        return -1
                    else:
                        return 0 
        # We have moved to the last sleep log record, only awake can reach this 
        return 0 
            
    def getSleepLogFromTS(self, startTS):
        ret = list() 
        for idx in range(len(self.sleepData)): 
            log = self.sleepData[idx]
            sleepTime = log.sleepTime 
            wakeupTime = log.wakeupTime 
            if wakeupTime is None:
                tmp = dict()
                tmp['start'] = sleepTime 
                tmp['end'] = -1
                tmp['duration'] = -1
                ret.append(tmp)
                continue 
            if startTS > wakeupTime:
                continue
            tmp = dict() 
            if idx == 0 and startTS > sleepTime:
                tmp['start'] = startTS 
            else:
                tmp['start'] = sleepTime 
            tmp['end'] = wakeupTime
            tmp['duration'] = ((wakeupTime - tmp['start']).total_seconds()) / 60 
            ret.append(tmp)
        return ret 




if __name__ == "__main__":      
    #dbName = "./data/6e44881f5af5d54a452b99f57899a7.db"    # Ke Huang
    #dbName = "./data/658ac828bdadbddaa909315ad80ac8.db"    # Xiang Ding
    #dbName = "./data/44c59ff1aa306a3490c52634c4bf76.db"    # Guanling Chen
    #dbName = "./data/78a4e254e6fa406ebe1bd3fab8a57499.db"    # NJU
    #dbName = "./data/82f1c1bd77582813b1de83b918c731b.db"    # Xu Ye 
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"   # Zhenyu Pan
    
    
    from glob import glob 
    dbList = glob('./data/*.db') 
    #dbList = ["./data/6e44881f5af5d54a452b99f57899a7.db"] 
    for dbName in dbList:
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        
        """
        for data in combiner.combinedDataList[:]:
            print "Time = " + str(data.createTime) + ", Screen On = " + str(data.screenOnSeconds) \
                + " isSleep = " + str(data.isSleep) 
        """
        
        
        from dataGenerator import DataGenerator
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0) 
        trainData, trainLabel = generator.generateTrainingDataSet(t = 15)
        testData, testLabel = generator.generateTestDataSet(t = 15) 
        fullData, fullLabel = generator.generateFullDataset(t = 15) 
        
        classifier = SleepClassifiers(fullData, fullLabel, trainData, trainLabel, testData, testLabel) 
        #classifier.chi2Test() 
        #classifier.fTest() 
        #classifier.featureRankingWithRFE() 
        
        """
        #predict1, score1 = classifier.SVMClassifier() 
        predict1, score1 = classifier.DTClassifier() 
        classifier.showClassificationInfo(predict1) 
        tmp = list()
        d0 = combiner.getSleepLogFromTS(generator.testCreateTimeList[0]) 
        tmp.append(d0) 
        for sleep in d0:
            print sleep 
        d1 = classifier.getSleepTimeAndDuration(testLabel, generator.testCreateTimeList)
        tmp.append(d1)
        print "" 
        for num in range(0, 6):
            smoothedPredicts = classifier.smoothPrediction(predict1, num) 
            classifier.showClassificationInfo(smoothedPredicts) 
            ret = classifier.getSleepTimeAndDuration(smoothedPredicts, generator.testCreateTimeList)
            tmp.append(ret)  
        from plotSet import plotSleepDuration 
        plotSleepDuration(tmp, generator.testCreateTimeList[0], generator.testCreateTimeList[-1], 0) 
        """
        
        #predict1, score1 = classifier.SVMClassifier() 
        #classifier.showClassificationInfo(predict1)
        #classifier.SVMClassifier(crossValidation = True)
        
        #predict2, score2 = classifier.NBClassifier()
        #classifier.showClassificationInfo(predict2) 
        #classifier.NBClassifier(crossValidation = True)
        
        #predict3, score3 = classifier.RFClassifier()
        #classifier.showClassificationInfo(predict3)
        #classifier.RFClassifier(crossValidation = True)
        
        #predict4, score4 = classifier.DTClassifier() 
        #classifier.showClassificationInfo(predict4)
        classifier.DTClassifier(crossValidation = True) 
        
        #predict5, score5 = classifier.LRClassifier() 
        #classifier.showClassificationInfo(predict5) 
        #classifier.LRClassifier(crossValidation = True)
        
        
        """
        finalPredict = list()
        for idx in range(len(testLabel)):
            if predict1[idx] + predict2[idx] + predict3[idx] + predict4[idx] + predict5[idx] >= 3:
                finalPredict.append(1)
            else:
                finalPredict.append(0) 
        print "\n\n====================  Combined classification Report"
        classifier.showClassificationInfo(finalPredict) 
        smoothPred = classifier.showSmoothedClassificationInfo(finalPredict) 
        """
        
        #for i in range(len(testLabel)):
        #    print str(generator.testCreateTimeList[i]) + ", GroundTruth = " + str(testLabel[i]) + ", Predict = " + str(smoothPred[i])
        