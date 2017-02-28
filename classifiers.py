def classifyNB(features_train, labels_train):   
    
    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    
    return clf

def classifySVM(features_train, labels_train):

	from sklearn import svm

	clf = svm.SVC()
	clf.fit(features_train, labels_train)

	return clf
