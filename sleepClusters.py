from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering 
from sklearn.cluster import Ward 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score  
from sklearn import metrics 
from sklearn.mixture import GMM, DPGMM 
from sklearn.hmm import MultinomialHMM, GaussianHMM
from time import time 
from sklearn.preprocessing import scale 
from dataCombiner import DataCombiner 
from dataGenerator import DataGenerator
import numpy as np

class SleepClusters(object):
    """
    A combination of a few cluster algorithms that used to cluster the
    sleep sensing data into two clusters. 
    """
    
    def __init__(self, combiner, t = 15):
        #self.combinedData = combiner.combinedDataList
        self.clusterNum = 2 
        self.target_names = ['Not_sleep', 'Sleep'] 
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.0) 
        self.data, self.label = generator.generateTrainingDataSet(t) 
        
    def adjustPredicts(self, predicts): 
        posCount = 0 
        posSum = 0.0 
        negCount = 0 
        negSum = 0.0 
        for idx in range(len(self.data)):
            if predicts[idx] == 1:
                posSum += self.data.item(idx, 5)
                posCount += 1
            else:
                negSum += self.data.item(idx, 5)
                negCount += 1 
        if (posSum / posCount) < (negSum / negCount):
            return predicts 
        else:
            return -(predicts - 1) 
        
    def bench_KMeans(self, method = 'k-means++'): 
        t0 = time()
        estimator = KMeans(n_clusters = self.clusterNum, init = method, n_jobs = -1, verbose = 0)
        data = scale(self.data)
        predicts = estimator.fit(data).labels_  
        print "\nKMeans Clustering Time: " + str(time() - t0) 
        #print estimator.cluster_centers_ 
        self.showClusteringMeasures(predicts) 
        return self.adjustPredicts(predicts) 
    
    def bench_spectral(self):
        t0 = time()
        estimator = SpectralClustering(n_clusters = self.clusterNum, 
                        eigen_solver = 'arpack', affinity = "nearest_neighbors") 
        data = scale(self.data)
        predicts = estimator.fit(data).labels_ 
        print "\nSpectral Clustering Time: " + str(time() - t0) 
        self.showClusteringMeasures(predicts) 
        return self.adjustPredicts(predicts)  
    
    def bench_ward(self):
        t0 = time() 
        estimator = Ward(n_clusters = self.clusterNum) 
        data = scale(self.data)
        predicts = estimator.fit(data).labels_ 
        print "\nWard hierarchical Clustering Time: " + str(time() - t0) 
        self.showClusteringMeasures(predicts) 
        return self.adjustPredicts(predicts) 
    
    def showClusteringMeasures(self, predicts):
        ARI = metrics.adjusted_rand_score(self.label, predicts) 
        #print "Adjusted Rand Index = " + str(round(ARI, 6)) 
        NMI = metrics.normalized_mutual_info_score(self.label, predicts)
        #print "Normalized Mutual Information (NMI) = " + str(round(NMI, 6)) 
        AMI = metrics.adjusted_mutual_info_score(self.label, predicts) 
        #print "Adjusted Mutual Information (AMI) = " + str(round(AMI, 6)) 
        #VMeasure = metrics.v_measure_score(self.label, predicts) 
        #print "V-measure = " + str(round(VMeasure, 6)) 
        print "%f | %f | %f | " % (ARI, NMI, AMI)
    
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
        
    # Un-supervised learning classification   
    def bench_GMM(self):
        print "\n================== Gaussian Mixture Model Classification " 
        t0 = time() 
        clf = GMM(n_components = 2, covariance_type = 'spherical') 
        clf.fit(self.data) 
        print "Weights = " + str(np.round(clf.weights_, 3)) 
        #print "Means = " + str(np.round(clf.means_, 3)) 
        #print "Covariance = " + str(np.round(clf.covars_, 3)) 
        predicts = clf.predict(self.data) 
        print "\nGMM processing Time: " + str(time() - t0) 
        return self.adjustPredicts(predicts) 
    
    def bench_DPGMM(self):
        print "\n================== Dirichlet Process Gaussian Mixture Model Classification " 
        t0 = time() 
        clf = DPGMM(n_components = 2, covariance_type = 'spherical') 
        clf.fit(self.data) 
        print "Weights = " + str(np.round(clf.weights_, 3)) 
        #print "Means = " + str(np.round(clf.means_, 3)) 
        #print "Covariance = " + str(np.round(clf.covars_, 3)) 
        predicts = clf.predict(self.data) 
        print "\nDPGMM processing Time: " + str(time() - t0) 
        return self.adjustPredicts(predicts) 
    
    def bench_HMM(self):
        print "\n================== Hidden Markov Model with multinomial (discrete) emissions" 
        t0 = time() 
        clf = GaussianHMM(n_components = 2) 
        clf.fit(self.data) 
        predicts = clf.predict(self.data) 
        print "\nDPGMM processing Time: " + str(time() - t0) 
        return self.adjustPredicts(predicts) 
    
    
    
    
    
    
if __name__ == "__main__":      
    #dbName = "./data/6e44881f5af5d54a452b99f57899a7.db"    # Ke Huang
    #dbName = "./data/658ac828bdadbddaa909315ad80ac8.db"    # Xiang Ding
    #dbName = "./data/44c59ff1aa306a3490c52634c4bf76.db"    # Guanling Chen
    #dbName = "./data/78a4e254e6fa406ebe1bd3fab8a57499.db"    # NJU
    #dbName = "./data/82f1c1bd77582813b1de83b918c731b.db"    # Xu Ye 
    #dbName = "./data/ecdbeb2bc610d72e726b58a04e463a3f.db"   # Zhenyu Pan
    
    from glob import glob 
    dbList = glob('./data/*.db') 
    #dbList = ['./data/6e44881f5af5d54a452b99f57899a7.db']
    for dbName in dbList:
        combiner = DataCombiner(dbName) 
        combiner.combineData()
        
        clf = SleepClusters(combiner, 12) 
        predicts = clf.bench_KMeans('k-means++') 
        clf.showClassificationInfo(predicts) 
        
        """
        predicts = clf.bench_spectral() 
        clf.showClassificationInfo(predicts) 
        """
        
        """
        predicts = clf.bench_ward() 
        clf.showClassificationInfo(predicts) 
        """
        
        """
        predicts = clf.bench_GMM() 
        clf.showClassificationInfo(predicts) 
        predicts = clf.bench_DPGMM() 
        clf.showClassificationInfo(predicts) 
        """
    
    