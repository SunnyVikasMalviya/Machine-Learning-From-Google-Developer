from sklearn import tree

#Data Set 
features = [[140, 20], [130, 20], [150, 30], [170, 30],\
            [145, 10], [155, 10], [155, 00], [165, 00]]
labels = ["2", "2", "3", "3", "0", "0", "1", "1"]
#Dataset collected

#Training a classifier
clf = tree.DecisionTreeClassifier()   #Creating a classifier
clf = clf.fit(features, labels)     #Training a classifier
#Classifier Trained

#Making Predictions
print(clf.predict([[150, 0]]))  #should print 1
print(clf.predict([[165, 10]])) #should print 0
#Predictions made


'''
    Glossary:
    
    SKLearn is the scikit library for machine learning.
    Classifiers : The Function that look at the training data and find pattern in them by themselves are called classifiers.
    Classifiers come under supervised learning. They take the data as the input and assigns a label to it as output.
    Supervised Learning : Technique to write the classifier automatically is called supervised learning.
    Algorithm : The method that the classifier uses to classify the data.
    Three steps:
    1. Collect Training Data
    2. Train the Classifier
    3. Make Predictions
    Feature : The features are the properties of the data that help the classifier find patterns in the data.
    Label : Labels are the classes that the input data is to be classified into.

    Working:

    Tree/Decision tree is an learning algorithm that the classifier clf is going to use.
    fit() is a training algorithm that trains the classifier with the features and the label.
    The input to the classifier is the features to new examples.
    scikit-learn uses real-valued features i.e. strings cannot be features when working with scikit-learn
'''
