from dataCombiner import DataCombiner 
from dataGenerator import DataGenerator 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from glob import glob 
import random 

class GeneralModel(object):
    def __init__(self, ratio = 0.2):
        self.ratio = ratio 
        self.target_names = ['Not_sleep', 'Sleep'] 
        self.dbList = self.getDBNames() 
        self.combindedData = self.getData() 
        generator = DataGenerator(self.combindedData, 0.0, 0.0) 
        self.data, self.label = generator.generateFullDataset() 
        self.clf = self.generalLRModel() 
        
    def getData(self): 
        data = list() 
        for dbFile in self.dbList: 
            combiner = DataCombiner(dbFile) 
            combiner.combineData() 
            totalLen = len(combiner.combinedDataList) 
            if totalLen < 10:
                continue 
            sampleNum = (int) (totalLen * self.ratio) 
            sampleIdxList = sorted(random.sample(range(totalLen), sampleNum)) 
            for idx in sampleIdxList:
                data.append(combiner.combinedDataList[idx]) 
        return data 
    
    def getDBNames(self): 
        files = glob('./data/*.db') 
        return files 
        
    def generalLRModel(self): 
        clf = LogisticRegression(C=0.01) 
        clf.fit(self.data, self.label) 
        return clf 
    
    def predictSleep(self, testData, testLabels): 
        predicts = self.clf.predict(testData) 
        self.showClassificationInfo(predicts, testLabels) 
        return predicts  
    
    def showClassificationInfo(self, predicts, labels):
        """
        Display the classification results 
        """
        accuracy = accuracy_score(labels, predicts) 
        print "\nClassification Accuracy: " + str(accuracy) 
        matrix = confusion_matrix(labels, predicts) 
        print "Confusion Matrix:" 
        print matrix 
        print "\n***** Classification Report *****"
        print (classification_report(labels, predicts, target_names = self.target_names)) 
    
if __name__ == "__main__": 
    """
    dbName = "./data/6e44881f5af5d54a452b99f57899a7.db"    # Ke Huang
    #dbName = "./data/658ac828bdadbddaa909315ad80ac8.db"    # Xiang Ding
    #dbName = "./data/44c59ff1aa306a3490c52634c4bf76.db"    # Guanling Chen
    #dbName = "./data/78a4e254e6fa406ebe1bd3fab8a57499.db"    # NJU
    #dbName = "./data/82f1c1bd77582813b1de83b918c731b.db"    # Xu Ye 
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"   # Zhenyu Pan
    
    combiner = DataCombiner(dbName) 
    combiner.combineData() 
    generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3) 
    fullData, fullLabel = generator.generateFullDataset() 
    """
    
    model = GeneralModel() 
    
    for dbFile in model.dbList:
        combiner = DataCombiner(dbFile) 
        combiner.combineData() 
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3) 
        fullData, fullLabel = generator.generateFullDataset() 
        print "\n\nDB: " + dbFile 
        model.predictSleep(fullData, fullLabel) 
    
    
    