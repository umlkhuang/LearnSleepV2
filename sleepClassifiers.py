from sklearn import svm 
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report, roc_curve, auc
from sklearn import cross_validation 
from sklearn.preprocessing import scale 
from sklearn.feature_selection import chi2, f_classif, RFE 
import pylab as pl 

class SleepClassifiers(object):
    """
    The classifier class for sleep classification. Multiple classification algorithms
    are included in this class. 
    """
    
    def __init__(self, data, label, trainData, trainLabel, testData, testLabel): 
        #self.data = scale(data)
        self.data = data 
        self.label = label 
        #self.trainData = scale(trainData) 
        self.trainData = trainData 
        #self.testData = scale(testData) 
        self.testData = testData 
        self.trainLabel = trainLabel 
        self.testLabel = testLabel 
        self.target_names = ['Not_sleep', 'Sleep'] 
    
    def chi2Test(self): 
        chi_val, p_val = chi2(self.data, self.label) 
        print "\nCHI values are: "
        print chi_val 
        print "\nP values are: "
        print p_val 
        return chi_val, p_val 
        
    def fTest(self):
        F, p_val = f_classif(self.data, self.label) 
        print "\nF values are: "
        print F 
        print "\nP values are: "
        print p_val 
        return F, p_val 
    
    def featureRankingWithRFE(self):
        clf = svm.LinearSVC(dual = False, class_weight = 'auto', verbose = 0) 
        #clf = LogisticRegression(C=0.01)
        selector = RFE(clf, 10, step = 1)
        selector = selector.fit(self.data, self.label) 
        print "\nREF support: "
        print selector.support_ 
        print "\nREF ranking: "
        print selector.ranking_ 
    
    def SVMClassifier(self, crossValidation = False, folds = 10):
        """
        SVM classifier 
        """
        print "\n================== SVM Classification " 
        clf = svm.LinearSVC(dual = False, class_weight = 'auto', verbose = 0) 
        if not crossValidation: 
            clf.fit(self.trainData, self.trainLabel) 
            predict = clf.predict(self.testData) 
            score = clf.decision_function(self.testData) 
            return predict, score 
        else:
            acc_scores = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'accuracy') 
            precisions = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'precision') 
            recalls = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'recall') 
            f1 = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'f1') 
            roc_aucs = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'roc_auc')
            ret = self.showCrossValidationInfo(acc_scores, precisions, recalls, f1, roc_aucs)
            return ret  
    
    def RFClassifier(self, seed = None, crossValidation = False, folds = 10):
        """
        Random Forest classifier 
        """

        print "\n================== Random Forest Classification "
        if seed is None:
            clf = RandomForestClassifier(n_jobs = -1, verbose = 0)
        else:
            clf = RandomForestClassifier(n_estimators=35, criterion='entropy', n_jobs = -1, random_state = seed)
        if not crossValidation:
            clf.fit(self.trainData, self.trainLabel) 
            predict = clf.predict(self.testData) 
            score = clf.predict_proba(self.testData)
            return predict, score[:, 1] 
        else:
            acc_scores = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'accuracy') 
            precisions = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'precision') 
            recalls = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'recall') 
            f1 = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'f1') 
            roc_aucs = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'roc_auc')
            ret = self.showCrossValidationInfo(acc_scores, precisions, recalls, f1, roc_aucs)
            return ret 
    
    def NBClassifier(self, crossValidation = False, folds = 10):
        """
        Naive Bayes Classifier (Gaussian)
        """
        print "\n================== Gaussian Naive Bayes Classification "
        clf = GaussianNB() 
        if not crossValidation:
            clf.fit(self.trainData, self.trainLabel) 
            predict = clf.predict(self.testData) 
            score = clf.predict_proba(self.testData)
            return predict, score[:, 1] 
        else:
            acc_scores = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'accuracy') 
            precisions = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'precision') 
            recalls = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'recall') 
            f1 = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'f1') 
            roc_aucs = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'roc_auc')
            ret = self.showCrossValidationInfo(acc_scores, precisions, recalls, f1, roc_aucs)
            return ret  
            
    
    def DTClassifier(self, crossValidation = False, folds = 10):
        """
        Decision Tree classifier 
        """
        print "\n================== Decision Tree Classification "
        clf = DecisionTreeClassifier(criterion = 'gini') 
        if not crossValidation:
            clf.fit(self.trainData, self.trainLabel) 
            predict = clf.predict(self.testData) 
            score = clf.predict_proba(self.testData)
            from sklearn import tree
            with open('model.dot', 'w') as f:
                f = tree.export_graphviz(clf, out_file = f) 
            return predict, score[:, 1] 
        else:
            acc_scores = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'accuracy') 
            precisions = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'precision') 
            recalls = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'recall') 
            f1 = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'f1') 
            roc_aucs = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'roc_auc')
            ret = self.showCrossValidationInfo(acc_scores, precisions, recalls, f1, roc_aucs)
            return ret 
    
    def LRClassifier(self, crossValidation = False, folds = 10):
        """
        Logistic Regression classifier 
        """
        print "\n================== Logistic Regression Classification "
        clf = LogisticRegression(C=0.01) 
        if not crossValidation:
            clf.fit(self.trainData, self.trainLabel) 
            predict = clf.predict(self.testData) 
            score = clf.decision_function(self.testData) 
            print clf.coef_ 
            return predict, score 
        else:
            acc_scores = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'accuracy') 
            precisions = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'precision') 
            recalls = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'recall') 
            f1 = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'f1') 
            roc_aucs = cross_validation.cross_val_score(clf, self.data, self.label, cv = folds, scoring = 'roc_auc')
            ret = self.showCrossValidationInfo(acc_scores, precisions, recalls, f1, roc_aucs)
            return ret  
    
    def plotROC(self, predict, score): 
        fpr, tpr, thresholds = roc_curve(predict, score, pos_label = 1) 
        roc_auc = auc(fpr, tpr) 
        pl.plot(fpr, tpr, lw = 2, label = "ROC area = %0.2f" % roc_auc) 
        pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck') 
        pl.xlim([-0.05, 1.05])
        pl.ylim([-0.05, 1.05])
        pl.xlabel('False Positive Rate')
        pl.ylabel('True Positive Rate')
        pl.title('Receiver operating characteristic')
        pl.legend(loc="lower right")
        pl.show() 
    
    def showCrossValidationInfo(self, acc_scores, precisions, recalls, f1, roc_aucs):
        print ""
        print "            Accuracy    Precision     Recall       F1   ROC_AUC" 
        print "" 
        ret = list() 
        folds = len(acc_scores) 
        for idx in range(folds): 
            """
            valueDict = dict() 
            valueDict['accuracy'] = acc_scores.item(idx) 
            valueDict['precision'] = precisions.item(idx) 
            valueDict['recall'] = recalls.item(idx) 
            valueDict['f1'] = f1.item(idx) 
            valueDict['roc'] = roc_aucs.item(idx) 
            ret.append(valueDict) 
            """
            print " Fold %2d      %4.3f       %4.3f       %4.3f     %4.3f     %4.3f" % (idx+1, acc_scores.item(idx), precisions.item(idx), recalls.item(idx), f1.item(idx), roc_aucs.item(idx)) 
        print ""
        print "     Avg      %4.3f       %4.3f       %4.3f     %4.3f     %4.3f" % (acc_scores.mean(), precisions.mean(), recalls.mean(), f1.mean(), roc_aucs.mean()) 
        print "     STD      %4.3f       %4.3f       %4.3f     %4.3f     %4.3f" % (acc_scores.std(), precisions.std(), recalls.std(), f1.std(), roc_aucs.std()) 
        print "" 
        avgDict = dict()
        avgDict['accuracy'] = acc_scores.mean() 
        avgDict['precision'] = precisions.mean() 
        avgDict['recall'] = recalls.mean() 
        avgDict['f1'] = f1.mean() 
        avgDict['roc'] = roc_aucs.mean() 
        ret.append(avgDict) 
        stdDict = dict()
        stdDict['accuracy'] = acc_scores.std() 
        stdDict['precision'] = precisions.std() 
        stdDict['recall'] = recalls.std() 
        stdDict['f1'] = f1.std() 
        stdDict['roc'] = roc_aucs.std() 
        ret.append(stdDict) 
        return ret 
            
    def showClassificationInfo(self, predict):
        """
        Display the classification results 
        """
        accuracy = accuracy_score(self.testLabel, predict) 
        f1 = f1_score(self.testLabel, predict) 
        print "Classification Accuracy: " + str(accuracy) 
        matrix = confusion_matrix(self.testLabel, predict) 
        print "Confusion Matrix:" 
        print matrix 
        print "\n***** Classification Report *****" 
        print (classification_report(self.testLabel, predict, target_names = self.target_names))
        return accuracy, f1
        
    def showSmoothedClassificationInfo(self, predict):
        """
        Display the smoothed classification results 
        """
        smoothPred = self.smoothPrediction(predict) 
        smoothAccuracy = accuracy_score(self.testLabel, smoothPred) 
        f1 = f1_score(self.testLabel, predict) 
        print "Smoothed Classification Accuracy: " + str(smoothAccuracy) 
        smoothMatrix = confusion_matrix(self.testLabel, smoothPred) 
        print "Smoothed Confusion Matrix:"
        print smoothMatrix 
        print "\n***** Smoothed Classification Report *****"
        print (classification_report(self.testLabel, smoothPred, target_names = self.target_names))
        return smoothAccuracy, f1 
    
    def smoothPrediction(self, rawPred, neighbor = 2):
        predict = list(rawPred) 
        num = neighbor 
        if neighbor == 0:
            return rawPred 
        #if neighbor < 2:
        #    num = 2
        sumVal = sum(predict[: (2 * num + 1)]) 
        for ptr in range(num, len(predict) - num):
            if sumVal > num:
                smoothVal = 1 
            else:
                smoothVal = 0 
            if predict[ptr] != smoothVal:
                sumVal = sumVal - predict[ptr] + smoothVal 
                predict[ptr] = smoothVal 
            sumVal -= predict[ptr - num] 
            if (ptr + num + 1) < len(predict):
                sumVal += predict[ptr + num + 1]
        return predict 
    
    def getSleepTimeAndDuration(self, predicts, createTimeList): 
        #print list(predicts) 
        ret = list() 
        firstValue = predicts[0] 
        testSize = len(predicts) 
        idx = 1 
        record = dict() 
        if firstValue == 1:
            record['start'] = createTimeList[0] 
        while idx < testSize:
            if predicts[idx] == predicts[idx - 1]: 
                interval = (createTimeList[idx] - createTimeList[idx - 1]).total_seconds() 
                if predicts[idx] == 1 and interval >= 3600 * 8:
                    record['end'] = createTimeList[idx - 1] 
                    record['duration'] = ((record['end'] - record['start']).total_seconds()) / 60 
                    ret.append(record) 
                    # New segment 
                    record = dict() 
                    record['start'] = createTimeList[idx] 
                # go to check the next record, increment idx by 1 
                idx += 1 
            else:
                # Previous record is sleep 
                if predicts[idx - 1] == 1:
                    record['end'] = createTimeList[idx - 1] 
                    record['duration'] = ((record['end'] - record['start']).total_seconds()) / 60 
                    ret.append(record) 
                # Previous record is awake
                else:
                    record = dict() 
                    record['start'] = createTimeList[idx] 
                idx += 1 
        return ret 
        
        """
        print list(predicts) 
        firstValue = predicts[0] 
        testSize = len(predicts) 
        startTS = 0 
        idx = 1 
        if firstValue == 1:
            while idx < testSize and predicts[idx] == 1:
                idx += 1 
        if idx < testSize:
            startTS = createTimeList[idx] 
        while idx < testSize and predicts[idx] == 0:
            idx += 1
        ret = list()
        while idx < testSize:
            record = dict() 
            record['start'] = createTimeList[idx] 
            while idx < testSize and predicts[idx] == 1: 
                idx += 1 
                if idx < testSize and (createTimeList[idx] - createTimeList[idx - 1]).total_seconds() >= 900:
                    break 
            if idx <= testSize:
                record['end'] = createTimeList[idx - 1] 
                record['duration'] = ((record['end'] - record['start']).total_seconds()) / 60
                ret.append(record) 
            while idx < testSize and predicts[idx] == 0:
                idx += 1 
        print startTS 
        for r in ret:
            print r 
        print "\n\n"
        return ret, startTS  
        """
        
        
        
        
        
    
    
    