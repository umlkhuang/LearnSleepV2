from glob import glob
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from dataCombiner import DataCombiner
from dataGenerator import DataGenerator
from util import Counter

if __name__ == "__main__":
    dbList = glob('./data/*.db')

    ret = dict()

    for dbFile in dbList:
        combiner = DataCombiner(dbFile)
        combiner.combineData()
        generator = DataGenerator(combiner.combinedDataList, 0.0, 0.3)
        fullData, fullLabel = generator.generateFullDataset(t = 12)
        print "=============  " + dbFile

        results = Counter()
        kf = cross_validation.KFold(len(fullData), n_folds = 10)
        for seed in range(0, 61):
            predicts = []
            for train_idx, test_idx in kf:
                X_train, X_test = fullData[train_idx], fullData[test_idx]
                Y_train, Y_test = fullLabel[train_idx], fullLabel[test_idx]
                clf = DecisionTreeClassifier(criterion = 'entropy', random_state=seed)
                #clf = RandomForestClassifier(n_estimators=35, criterion='entropy', n_jobs=-1, random_state=seed)
                clf.fit(X_train, Y_train)
                predicts += list(clf.predict(X_test))
                del X_train
                del X_test
                del Y_train
                del Y_test
                del clf
            accuracy = accuracy_score(fullLabel, predicts)
            print "\tSeed = %d, accuracy = %f" % (seed, accuracy)
            results[seed] = accuracy
            del predicts
        print ""
        ret[dbFile] = results.argMax()
        del combiner
        del generator
        del fullData
        del fullLabel

    for dbFile in ret.keys():
        print "\"%s\": %d" % (dbFile, ret[dbFile])

