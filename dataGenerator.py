from numpy import array 
from util import Counter 
from constants import MINIMUM_TIMES_WIFI_APPEAR 

class DataGenerator(object):
    """
    The class that used to generate Numpy array for the learning task. 
    The input is a list of combined data. 
    """
    def __init__(self, combinedDataList, validationRatio = 0.2, testRatio = 0.2):
        self.raw = combinedDataList 
        self.validationRatio = validationRatio 
        self.testRatio = testRatio 
        self.totalDataLen = len(combinedDataList) 
        self.trainNum = (int)(self.totalDataLen * (1 - self.testRatio - self.validationRatio))
        self.validationNum = (int)(self.totalDataLen * self.validationRatio) 
        self.testNum = self.totalDataLen - self.trainNum - self.validationNum 
        self.appUsageList = self.getAppUsageList() 
        self.wifiList = self.getWifiList() 
        self.fullCreateTimeList = list() 
        self.trainCreateTimeList = list() 
        self.validationCreateTimeList = list()
        self.testCreateTimeList = list() 
        print "[2] In data generator! " 
        print "Total sensing data after filtering = " + str(len(self.raw)) 
        print "No. of WiFi features = " + str(len(self.wifiList))
        print "No. of App usage features = " + str(len(self.appUsageList)) 
    
    def getAppUsageList(self):
        appUsageSet = set() 
        for idx in range(0, self.trainNum):
            usageCounter = self.raw[idx].appUsage 
            for key in usageCounter.keys():
                appUsageSet.add(key) 
        return list(appUsageSet) 
    
    def getWifiList(self):
        wifiSet = set() 
        totalWifiCounter = Counter() 
        for idx in range(0, self.trainNum):
            wifiCounter = self.raw[idx].wifiCounter
            for key in wifiCounter.keys():
                totalWifiCounter[key] += 1 
        for key in totalWifiCounter:
            if totalWifiCounter[key] >= MINIMUM_TIMES_WIFI_APPEAR:
                wifiSet.add(key) 
        return list(wifiSet) 
    
    def generateDataList(self, combinedData, t = 15):
        """
        Generate the final learning data array and label from the combined data instance 
        """
        data = list() 
        if t / 8 >= 1:
            #data.append(combinedData.month)
            data.append(combinedData.day)
            data.append(combinedData.dayOfWeek)
            data.append(combinedData.hour) 
        if (t % 8) / 4 >= 1:
            data.append(combinedData.movement)
            data.append(combinedData.illuminanceMax)
            data.append(combinedData.illuminanceMin)
            data.append(combinedData.illuminanceAvg)
            data.append(combinedData.illuminanceStd)
            data.append(combinedData.decibelMax)
            data.append(combinedData.decibelMin)
            data.append(combinedData.decibelAvg)
            data.append(combinedData.decibelStd)
            data.append(combinedData.isCharging) 
            data.append(combinedData.powerLevel)
            data.append(combinedData.proximity) 
            data.append(combinedData.screenOnSeconds) 
        
        if (t % 4) / 2 >= 1:
            for i in range(0, len(self.wifiList)): 
                data.append(combinedData.wifiCounter[self.wifiList[i]])
        if (t % 2) == 1:
            for i in range(0, len(self.appUsageList)):
                data.append(combinedData.appUsage[self.appUsageList[i]] > 0)
        
        return data 
    
    def generateFullDataset(self, t = 15):
        data = list() 
        label = list() 
        for idx in range(len(self.raw)): 
            combinedData = self.raw[idx] 
            data.append(self.generateDataList(combinedData, t)) 
            label.append(combinedData.isSleep) 
            self.fullCreateTimeList.append(combinedData.createTime) 
        return array(data), array(label) 
    
    def generateTrainingDataSet(self, t = 15): 
        data = list() 
        label = list() 
        for idx in range(0, self.trainNum):
            combinedData = self.raw[idx] 
            data.append(self.generateDataList(combinedData, t))
            label.append(combinedData.isSleep) 
            self.trainCreateTimeList.append(combinedData.createTime) 
        return array(data), array(label) 
    
    def generateValidationDataSet(self, t = 15): 
        data = list() 
        label = list() 
        for idx in range(self.trainNum, self.trainNum + self.validationNum): 
            combinedData = self.raw[idx] 
            data.append(self.generateDataList(combinedData, t))
            label.append(combinedData.isSleep) 
            self.validationCreateTimeList.append(combinedData.createTime)
        return array(data), array(label) 
    
    def generateTestDataSet(self, t = 15):
        data = list() 
        label = list() 
        for combinedData in self.raw[(self.trainNum + self.validationNum):]:
            data.append(self.generateDataList(combinedData, t))
            label.append(combinedData.isSleep) 
            self.testCreateTimeList.append(combinedData.createTime) 
        return array(data), array(label) 
    
    
        
    
    
    
    