'''
Dataset Used : Iris flower data set
https://en.wikipedia.org/wiki/Iris_flower_data_set
Four Features, 150 datas, 3 types of flowers namely Sertosa Virginica Versicolor
Lables are used to represent the flowers: 0:sertosa, 1:virginica, 2:versicolor
Steps :
1. Import Dataset
2. Train classifier
3. Predict lable of the flower
Scikit learn module already contains few datasets including the iris data set.
http://scikit-learn.org/stable/datasets/index.html
The dataset contains both the table of data as well as the meta data.
Meta data tells the features being used and the distinct labels for the features
'''
import numpy as np
from sklearn import tree

#importing the inbuilt iris dataset from sklearn library
from sklearn.datasets import load_iris as lr
iris = lr()

#Displaying the features used for the problem
print(iris.feature_names)
#Displaying the lables for the datas
#Since scikit learn only takes numeral inputs, sertosa is labled as  0, virginica as 1 and versicolor as 2
print(iris.target_names)

#Accessing the different values from the dataset 
#0th index data is for sertosa, 50th index data is for virginica and 100th index data is for versicolor
print(iris.data[0])
print(iris.target[0])
print(iris.data[50])
print(iris.target[50])
print(iris.data[100])
print(iris.target[100])

#Displaying the whole dataset
for i in range(len(iris.target)):
    print("Example %d : label %s, features %s" % (i, iris.target[i], iris.data[i]))

#Actual program starts here
#Below variable stores the indices of the different testing datas we will use
test_index = [0, 27, 50, 33, 112, 100]

#Training Data
#Deleting the testing data from the whole dataset
train_target = np.delete(iris.target, test_index)
train_data = np.delete(iris.data, test_index, axis = 0)

#Testing Data
#Storing the data and lables of the testing data in variables
test_target = iris.target[test_index]
test_data = iris.data[test_index]

#Training a classifier
clf = tree.DecisionTreeClassifier()     #Algorithm used is Decision Tree
clf.fit(train_data, train_target)       #Trainig Step

#Predicting label for new flower
print(test_target)                      #Required output
print(clf.predict(test_data))           #Predicted output

'''
Note:
Give some extra time to select the features that you are going to use becauese it matters a lot.
'''
