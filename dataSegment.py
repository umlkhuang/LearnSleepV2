from numpy import array 
from dataCombiner import DataCombiner 
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering 
from sklearn.cluster import Ward 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score 
from sklearn import metrics 
from time import time 
from sklearn.preprocessing import scale 
from glob import glob 
from datetime import timedelta 

class DataSegment(object):
    def __init__(self, combinedData, testRatio = 0.3):
        self.combinedData = combinedData 
        self.totalLen = len(self.combinedData)
        self.testRatio = testRatio  
        if testRatio == 0.0:
            self.breakId = 0
        else:
            self.breakId = int(self.totalLen * (1 - testRatio)) 
        self.continueDataBreakId = 0  
        self.createTimeList = self.getCreateTimeList() 
        if testRatio == 0.0:
            self.breakTime = self.createTimeList[-1]
        else:
            self.breakTime = self.createTimeList[self.breakId] 
        self.target_names = ['Not_sleep', 'Sleep'] 
        self.HOUR_BUCKETS = 3 
        self.MAXMOVEMENT = 6
        self.data, self.label, self.timeList = self.segmentData() 
        
    def getCreateTimeList(self):
        ret = list()
        for instance in self.combinedData:
            ret.append(instance.createTime) 
        return ret 
    
    def getSleepTimeAndDuration(self, predicts):
        testLabel = predicts[self.continueDataBreakId:]
        testTime = self.timeList[self.continueDataBreakId:] 
        ret = list() 
        firstValue = testLabel[0] 
        testSize = len(testLabel) 
        idx = 1 
        record = dict() 
        if firstValue == 1:
            record['start'] = testTime[0]['start'] 
            record['end'] = testTime[0]['end'] 
        while idx < testSize:
            if testLabel[idx] == 1 and testLabel[idx] == testLabel[idx - 1]:
                interval = (testTime[idx]['start'] - testTime[idx - 1]['end']).total_seconds() 
                if testLabel[idx] == 1 and interval >= 3600 * 8: 
                    record['end'] = testTime[idx - 1]['end'] 
                    record['duration'] = ((record['end'] - record['start']).total_seconds()) / 60
                    #if record['duration'] >= 120: 
                    #    ret.append(record)
                    ret.append(record)
                    # New segment 
                    record = dict()
                    record['start'] = testTime[idx]['start']
                    record['end'] = testTime[idx]['end']
                else:
                    record['end'] = testTime[idx]['end']
                idx += 1
            else:
                # Previous record is sleep 
                if testLabel[idx - 1] == 1:
                    record['end'] = testTime[idx - 1]['end'] 
                    record['duration'] = ((record['end'] - record['start']).total_seconds()) / 60 
                    if record['duration'] >= 120:
                        ret.append(record) 
                    #ret.append(record)
                # Previous record is awake
                else:
                    record = dict() 
                    record['start'] = testTime[idx]['start']
                    record['end'] = testTime[idx]['end'] 
                idx += 1
        #for r in ret:
        #    print r 
        print "\n"
        return ret 
    
    def segmentData(self):
        data = list()
        label = list()
        timeList = list() 
        tmpData = list() 
        tmpTime = dict() 
        hourBuckets = [0] * self.HOUR_BUCKETS 
        lightMaxList = list()
        lightMinList = list()
        lightAvgList = list()
        decibelMaxList = list()
        decibelMinList = list()
        decibelAvgList = list()
        powerList = list()
        proximityList = list()
        screenOnList = list() 
        labelList = list() 
        firstRecord = self.combinedData[0] 
        if firstRecord.movement <= self.MAXMOVEMENT:
            startTime = firstRecord.createTime 
            endTime = firstRecord.createTime + timedelta(0, 300) 
            tmpTime['start'] = startTime
            tmpTime['end'] = endTime 
            hourBuckets[firstRecord.hour / (24 / self.HOUR_BUCKETS)] += 1.0 
            lightMaxList.append(firstRecord.illuminanceMax) 
            lightMinList.append(firstRecord.illuminanceMin)
            lightAvgList.append(firstRecord.illuminanceAvg) 
            decibelMaxList.append(firstRecord.decibelMax)
            decibelMinList.append(firstRecord.decibelMin)
            decibelAvgList.append(firstRecord.decibelAvg)
            powerList.append(firstRecord.powerLevel) 
            proximityList.append(firstRecord.proximity)
            screenOnList.append(firstRecord.screenOnSeconds) 
            labelList.append(firstRecord.isSleep) 
        for idx in range(1, len(self.combinedData)): 
            instance = self.combinedData[idx] 
            createTime = instance.createTime
            movement = instance.movement 
            if movement <= self.MAXMOVEMENT:
                # The phone is still, check if this instance should be combined to previous segment 
                if self.combinedData[idx - 1].movement <= self.MAXMOVEMENT: 
                    if idx == self.breakId:
                        self.continueDataBreakId = len(data) + 1 
                    # Check the time delta to the previous record 
                    if (createTime - self.combinedData[idx - 1].createTime).total_seconds() >= 360:
                        
                        # split as a new segment 
                        bucketCnt = sum(hourBuckets)
                        hourBuckets = [val / bucketCnt for val in hourBuckets] 
                        tmpData += hourBuckets      # Append the hour buckets to data list 
                        #tmpData.append(array(lightMaxList).mean())      # Append the Max illuminance mean 
                        #tmpData.append(array(lightMaxList).std())       # Append the Max illuminance std 
                        #tmpData.append(array(lightMinList).mean())      # Append the Min illuminance mean 
                        #tmpData.append(array(lightMinList).std())       # Append the Min illuminance std 
                        tmpData.append(array(lightAvgList).mean())      # Append the Avg illuminance mean 
                        tmpData.append(array(lightAvgList).std())       # Append the Avg illuminance std 
                        #tmpData.append(array(decibelMaxList).mean())    # Append the Max decibel mean 
                        #tmpData.append(array(decibelMaxList).std())     # Append the Max decibel std 
                        #tmpData.append(array(decibelMinList).mean())    # Append the Min decibel mean 
                        #tmpData.append(array(decibelMinList).std())     # Append the Min decibel std 
                        tmpData.append(array(decibelAvgList).mean())    # Append the Avg decibel mean 
                        tmpData.append(array(decibelAvgList).std())     # Append the Avg decibel std 
                        tmpData.append(array(powerList).mean())         # Append the power level mean 
                        tmpData.append(array(powerList).std())          # Append the power level std 
                        #tmpData.append(array(proximityList).mean())     # Append the proximity mean 
                        #tmpData.append(array(proximityList).std())      # Append the proximity std 
                        tmpData.append(array(screenOnList).mean())      # Append the screen on mean 
                        tmpData.append(array(screenOnList).std())       # Append the screen on std 
                        #if len(timeList) >= 1:
                        #    tmpData.append((tmpTime['start'] - timeList[-1]['end']).total_seconds())
                        #else:
                        #    tmpData.append(0)
                        tmpData.append((tmpTime['end'] - tmpTime['start']).total_seconds())
                        # Append the data record to the final data and label lists 
                        data.append(tmpData)
                        timeList.append(tmpTime) 
                        if sum(labelList) * 2 < len(labelList):
                            label.append(0)
                        else:
                            label.append(1) 
                        
                        # Initialize everything as a new segment 
                        tmpData = list() 
                        tmpTime = dict() 
                        hourBuckets = [0] * self.HOUR_BUCKETS 
                        lightMaxList = list()
                        lightMinList = list()
                        lightAvgList = list()
                        decibelMaxList = list()
                        decibelMinList = list()
                        decibelAvgList = list()
                        powerList = list()
                        proximityList = list()
                        screenOnList = list() 
                        labelList = list() 
                        tmpTime['start'] = createTime 
                        tmpTime['end'] = createTime + timedelta(0, 300) 
                        hourBuckets[instance.hour / (24 / self.HOUR_BUCKETS)] += 1.0
                        lightMaxList.append(instance.illuminanceMax) 
                        lightMinList.append(instance.illuminanceMin)
                        lightAvgList.append(instance.illuminanceAvg) 
                        decibelMaxList.append(instance.decibelMax)
                        decibelMinList.append(instance.decibelMin)
                        decibelAvgList.append(instance.decibelAvg)
                        powerList.append(instance.powerLevel) 
                        proximityList.append(instance.proximity)
                        screenOnList.append(instance.screenOnSeconds) 
                        labelList.append(instance.isSleep) 
                        
                    else:
                        # Combine to the previous segment 
                        tmpTime['end'] = createTime 
                        hourBuckets[instance.hour / (24 / self.HOUR_BUCKETS)] += 1.0
                        lightMaxList.append(instance.illuminanceMax) 
                        lightMinList.append(instance.illuminanceMin)
                        lightAvgList.append(instance.illuminanceAvg) 
                        decibelMaxList.append(instance.decibelMax)
                        decibelMinList.append(instance.decibelMin)
                        decibelAvgList.append(instance.decibelAvg)
                        powerList.append(instance.powerLevel) 
                        proximityList.append(instance.proximity)
                        screenOnList.append(instance.screenOnSeconds) 
                        labelList.append(instance.isSleep) 
                # Current instance is till, but previous instance is moving 
                else:
                    # This is a new segment 
                    if idx == self.breakId:
                        self.continueDataBreakId = len(data) 
                    
                    # Initialize everything again 
                    tmpData = list() 
                    tmpTime = dict() 
                    hourBuckets = [0] * self.HOUR_BUCKETS 
                    lightMaxList = list()
                    lightMinList = list()
                    lightAvgList = list()
                    decibelMaxList = list()
                    decibelMinList = list()
                    decibelAvgList = list()
                    powerList = list()
                    proximityList = list()
                    screenOnList = list() 
                    labelList = list() 
                    tmpTime['start'] = createTime 
                    tmpTime['end'] = createTime + timedelta(0, 300) 
                    hourBuckets[instance.hour / (24 / self.HOUR_BUCKETS)] += 1.0
                    lightMaxList.append(instance.illuminanceMax) 
                    lightMinList.append(instance.illuminanceMin)
                    lightAvgList.append(instance.illuminanceAvg) 
                    decibelMaxList.append(instance.decibelMax)
                    decibelMinList.append(instance.decibelMin)
                    decibelAvgList.append(instance.decibelAvg)
                    powerList.append(instance.powerLevel) 
                    proximityList.append(instance.proximity)
                    screenOnList.append(instance.screenOnSeconds) 
                    labelList.append(instance.isSleep) 
            else:
                # The phone moved, check if we need to add segment data into list 
                if self.combinedData[idx - 1].movement > self.MAXMOVEMENT:
                    # Previous record is also not still 
                    if idx == self.breakId:
                        self.continueDataBreakId = len(data) + 1
                    continue 
                else:
                    if idx == self.breakId:
                        self.continueDataBreakId = len(data) + 1
                    
                    bucketCnt = sum(hourBuckets)
                    hourBuckets = [val / bucketCnt for val in hourBuckets] 
                    tmpData += hourBuckets      # Append the hour buckets to data list 
                    #tmpData.append(array(lightMaxList).mean())      # Append the Max illuminance mean 
                    #tmpData.append(array(lightMaxList).std())       # Append the Max illuminance std 
                    #tmpData.append(array(lightMinList).mean())      # Append the Min illuminance mean 
                    #tmpData.append(array(lightMinList).std())       # Append the Min illuminance std 
                    tmpData.append(array(lightAvgList).mean())      # Append the Avg illuminance mean 
                    tmpData.append(array(lightAvgList).std())       # Append the Avg illuminance std 
                    #tmpData.append(array(decibelMaxList).mean())    # Append the Max decibel mean 
                    #tmpData.append(array(decibelMaxList).std())     # Append the Max decibel std 
                    #tmpData.append(array(decibelMinList).mean())    # Append the Min decibel mean 
                    #tmpData.append(array(decibelMinList).std())     # Append the Min decibel std 
                    tmpData.append(array(decibelAvgList).mean())    # Append the Avg decibel mean 
                    tmpData.append(array(decibelAvgList).std())     # Append the Avg decibel std 
                    tmpData.append(array(powerList).mean())         # Append the power level mean 
                    tmpData.append(array(powerList).std())          # Append the power level std 
                    #tmpData.append(array(proximityList).mean())     # Append the proximity mean 
                    #tmpData.append(array(proximityList).std())      # Append the proximity std 
                    tmpData.append(array(screenOnList).mean())      # Append the screen on mean 
                    tmpData.append(array(screenOnList).std())       # Append the screen on std 
                    #if len(timeList) >= 1:
                    #    tmpData.append((tmpTime['start'] - timeList[-1]['end']).total_seconds())
                    #else:
                    #    tmpData.append(0)
                    tmpData.append((tmpTime['end'] - tmpTime['start']).total_seconds())
                    # Append the data record to the final data and label lists 
                    data.append(tmpData)
                    timeList.append(tmpTime) 
                    if sum(labelList) * 2 < len(labelList):
                        label.append(0)
                    else:
                        label.append(1) 
                    
                    # Clean the lightMaxList as a indication of added the data 
                    lightMaxList = list()
        if len(lightMaxList) > 0:
            # Need to add the last segment 
            bucketCnt = sum(hourBuckets)
            hourBuckets = [val / bucketCnt for val in hourBuckets] 
            tmpData += hourBuckets      # Append the hour buckets to data list 
            #tmpData.append(array(lightMaxList).mean())      # Append the Max illuminance mean 
            #tmpData.append(array(lightMaxList).std())       # Append the Max illuminance std 
            #tmpData.append(array(lightMinList).mean())      # Append the Min illuminance mean 
            #tmpData.append(array(lightMinList).std())       # Append the Min illuminance std 
            tmpData.append(array(lightAvgList).mean())      # Append the Avg illuminance mean 
            tmpData.append(array(lightAvgList).std())       # Append the Avg illuminance std 
            #tmpData.append(array(decibelMaxList).mean())    # Append the Max decibel mean 
            #tmpData.append(array(decibelMaxList).std())     # Append the Max decibel std 
            #tmpData.append(array(decibelMinList).mean())    # Append the Min decibel mean 
            #tmpData.append(array(decibelMinList).std())     # Append the Min decibel std 
            tmpData.append(array(decibelAvgList).mean())    # Append the Avg decibel mean 
            tmpData.append(array(decibelAvgList).std())     # Append the Avg decibel std 
            tmpData.append(array(powerList).mean())         # Append the power level mean 
            tmpData.append(array(powerList).std())          # Append the power level std 
            #tmpData.append(array(proximityList).mean())     # Append the proximity mean 
            #tmpData.append(array(proximityList).std())      # Append the proximity std 
            tmpData.append(array(screenOnList).mean())      # Append the screen on mean 
            tmpData.append(array(screenOnList).std())       # Append the screen on std 
            #if len(timeList) >= 1:
            #    tmpData.append((tmpTime['start'] - timeList[-1]['end']).total_seconds())
            #else:
            #    tmpData.append(0)
            tmpData.append((tmpTime['end'] - tmpTime['start']).total_seconds())
            # Append the data record to the final data and label lists 
            data.append(tmpData) 
            timeList.append(tmpTime) 
            if sum(labelList) * 2 < len(labelList):
                label.append(0)
            else:
                label.append(1) 
        return array(data), array(label), timeList 
            
    def newAdjustPredicts(self, predicts):
        posCount = 0 
        posSum = 0.0 
        negCount = 0 
        negSum = 0.0 
        for idx in range(len(self.data)):
            if predicts[idx] == 1:
                posSum += self.data.item(idx, self.HOUR_BUCKETS) 
                posCount += 1
            else:
                negSum += self.data.item(idx, self.HOUR_BUCKETS)
                negCount += 1
        if (posSum / posCount) < (negSum / negCount):
            return predicts 
        else:
            return -(predicts - 1) 
        
    
    def showClusteringMeasures(self, predicts):
        ARI = metrics.adjusted_rand_score(self.label, predicts) 
        #print "Adjusted Rand Index = " + str(round(ARI, 6)) 
        NMI = metrics.normalized_mutual_info_score(self.label, predicts)
        #print "Normalized Mutual Information (NMI) = " + str(round(NMI, 6)) 
        AMI = metrics.adjusted_mutual_info_score(self.label, predicts) 
        #print "Adjusted Mutual Information (AMI) = " + str(round(AMI, 6)) 
        #VMeasure = metrics.v_measure_score(self.label, predicts) 
        #print "V-measure = " + str(round(VMeasure, 6)) 
        print "%f | %f | %f | " % (ARI, NMI, AMI), 
        
    def showClassificationInfo(self, predict):
        """
        Display the clustering results 
        """
        accuracy = accuracy_score(self.label, predict) 
        print "\nClustering Accuracy: " + str(round(accuracy, 4)) 
        f1 = f1_score(self.label, predict) 
        print "F1 score: " + str(round(f1, 4))
        matrix = confusion_matrix(self.label, predict) 
        print "Confusion Matrix:" 
        print matrix 
        print "\n***** Classification Report *****" 
        print (classification_report(self.label, predict, target_names = self.target_names))
        return accuracy, f1 
    
if __name__ == "__main__": 
    dbName = "./data/6e44881f5af5d54a452b99f57899a7.db"    # Ke Huang
    #dbName = "./data/658ac828bdadbddaa909315ad80ac8.db"    # Xiang Ding
    #dbName = "./data/44c59ff1aa306a3490c52634c4bf76.db"    # Guanling Chen
    #dbName = "./data/78a4e254e6fa406ebe1bd3fab8a57499.db"    # NJU
    #dbName = "./data/82f1c1bd77582813b1de83b918c731b.db"    # Xu Ye 
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"   # Zhenyu Pan 
    
    #dbList = glob('./data/*.db') 
    dbList = ['./data/18dcdfbc751064e9251fa718a9319fe6.db'] 
    #dbList = [dbName] 
    
    for dbName in dbList:
        combiner = DataCombiner(dbName) 
        combiner.combineData() 
        segmenter = DataSegment(combiner.combinedDataList, 0.0) 
        
        #print len(segmenter.data), segmenter.continueDataBreakId 
        #ret = segmenter.getSleepTimeAndDuration(segmenter.label) 
        
        #for item in ret:
        #    print item 
        
        for idx in range(len(segmenter.timeList)):
            print str(segmenter.label[idx]) + str(segmenter.timeList[idx])
        
        
        t0 = time()
        estimator = KMeans(n_clusters = 2, init = 'k-means++', n_jobs = -1, verbose = 0)
        #estimator = SpectralClustering(n_clusters = 2, eigen_solver = 'arpack', affinity = "nearest_neighbors")
        #estimator = Ward(n_clusters = 2)  
        data = scale(segmenter.data)
        predicts = estimator.fit(data).labels_  
        print "\nClustering Time: " + str(time() - t0) 
        segmenter.showClusteringMeasures(predicts) 
        #print "Ground Truth:"
        #print segmenter.label.tolist()  
        #print "Raw predictions:" 
        #print predicts.tolist()
        predicts = segmenter.newAdjustPredicts(predicts) 
        #predicts = segmenter.adjustPredicts(predicts) 
        #print "Adjusted predictions:" 
        #print predicts.tolist() 
        segmenter.showClassificationInfo(predicts) 
        print "\n"
        

