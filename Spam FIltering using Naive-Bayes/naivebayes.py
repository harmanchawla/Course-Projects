
''' The code is written in Python 3.7. 

In order to run the code, simply run naivebayes.py (python3 filename.py)
Before you run this, make sure you have the spamtrain.csv and spamtest.csv files in the same directory 
Output: 1 is SPAM and 0 is HAM '''

''' Note: I have only used Pandas to import the CSV file. The gen function found in numpy
was not importing the first line of data for some reason even though header was set to None and 
Names left undefined. I later convert them to numpy arrays.'''

import numpy as np 
import math 
import pandas as pd

''' Importing the csv files (train and test) and storing them as 2D arrays '''
trainset = pd.read_csv('spambasetrain.csv', sep=',', header=None)
testset = pd.read_csv('spambasetest.csv', sep=',', header=None)

# trainset= np.genfromtxt('spambasetrain.csv',delimiter=',',dtype=None)
# testset = np.genfromtxt('spambasetest.csv', delimiter=',', dtype=None)

trainset = trainset.values
testset = testset.values

''' Intitalizing values to count the number of instances of spam and ham in the training set '''
totalTrainSet = len(trainset)
totalSpam = 0
totalHam = 0

''' Choosing not to use the numpy function count_nonzero() '''
for x in range(totalTrainSet):
	if trainset[x][9] == 1:
		totalSpam = totalSpam + 1
	else:
		totalHam += 1

''' Step 1: Calculating P(Ham) and P(Spam) '''
probabilitySpam = totalSpam/totalTrainSet
probabilityHam = totalHam/totalTrainSet

''' Initializing arrays to store mean and standard deviation for each attribute. '''
meanSpam = np.zeros(9)
varianceSpam = np.zeros(9)
meanHam = np.zeros(9)
varianceHam = np.zeros(9)

rowLimit = totalTrainSet
columnLimit = len(trainset[1]) - 1

tempSumClass1 = 0
tempSumClass2 = 0

''' Calculating mean for each attribute and adding it to the respective list. '''
for j in range(columnLimit):
	for i in range(rowLimit):
		# check if it is a spam or a ham
		if trainset[i][9] == 1:
			tempSumClass1 += trainset[i][j]
		elif trainset[i][9] == 0: 
			tempSumClass2 += trainset[i][j]

	tempSumClass1 = tempSumClass1/totalSpam
	tempSumClass2 = tempSumClass2/totalHam
	meanSpam[j] = tempSumClass1
	meanHam[j] = tempSumClass2


'''  Calculating variance for each attribute and adding it to the respective list. '''
tempSumClass1 = 0
tempSumClass2 = 0
mean = 0

for j in range(columnLimit):
	for i in range(rowLimit):
		if trainset[i][9] == 1:
			mean = meanSpam[j]
			tempSumClass1 += math.pow((trainset[i][j] - mean), 2)
		elif trainset[i][9] == 0:
			mean = meanHam[j]
			tempSumClass2 += math.pow((trainset[i][j] - mean), 2)

	tempSumClass1 = tempSumClass1/(totalSpam - 1)
	tempSumClass2 = tempSumClass2/(totalHam - 1)
	varianceSpam[j] = tempSumClass1
	varianceHam[j] = tempSumClass2

# Define probability model for spam
def probabilityClassSpam(index):
	columnLimit = len(testset[index]) - 1
	suspectProb = sum(
		probabilityDistribution(testset[index][i], meanSpam[i], varianceSpam[i])
		for i in range(columnLimit)
	)
	suspectProb += math.log(probabilitySpam)
	return suspectProb

# Define probability model for ham
def probabilityClassHam(index):
	columnLimit = len(testset[index]) - 1
	suspectProb = sum(
		probabilityDistribution(testset[index][i], meanHam[i], varianceHam[i])
		for i in range(columnLimit)
	)
	suspectProb += math.log(probabilityHam)
	return suspectProb

# Calculating probability 
def probabilityDistribution(x, mean, variance):
	denominator = math.sqrt(2*(math.pi)*variance)
	power = -1*((x - mean)*(x - mean))/(2*variance)
	numerator = math.pow((math.e), power)

	probability = numerator/denominator
	probability = math.log(probability)
	return probability

''' Calculating probability for each test case. '''

totalTestSet = len(testset)
rowLimit = totalTestSet
columnLimit = len(testset[1]) - 1
probabilityClass1 = 0
probabilityClass2 = 0
prediction = []

for i in range(rowLimit):
	probabilityClass1 = probabilityClassSpam(i)
	probabilityClass2 = probabilityClassHam(i)

	if probabilityClass1 > probabilityClass2:
		prediction.append(1)
	else:
		prediction.append(0)


''' Computing Accurary '''
count = sum(1 for i in range(rowLimit) if prediction[i] == testset[i][9])
# Answers
print('Probability for Class Spam: ', probabilitySpam)
print('Probability for Class Ham: ', probabilityHam)
print('Means for attributes for Class Spam: ', meanSpam)
print('Means for attributes for Class Ham: ', meanHam)
print('Variance for attributes for Class Spam: ', varianceSpam)
print('Variance for attributes for Class Ham: ', varianceHam)
print('Predicted classes for all test examples: ', prediction)
print('Correctly classified: ', count, 'Total: ', rowLimit, '\n')
print('Incorrect Predictions: ', rowLimit - count)
print('Percentage error: ', 100 - 100*count / rowLimit, '\n')
print('Accurary: ', 100*count / rowLimit)
#question 10
zero_r = np.zeros(200)
count = sum(1 for i in range(200) if testset[i][9] == zero_r[i])
count *= 100
print('Answer 10 (Zero-R): ', count/200)

