import collections
import math
import numpy
from random import randint

def loadData(fileName, features=[], labels=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        currentLine = []
        for feature in line[:-1]:
            currentLine.append(float(feature))
        features.append(currentLine)
        labels.append(line[len(line)-1])

def regularPerceptron(trainingSet, trainingLabels, w1):
    for i in range(len(trainingSet)):
        dotProduct = numpy.dot(numpy.array(trainingSet[i]), w1)
        if float(trainingLabels[i] ) * float(dotProduct) <= 0:
            w1 = numpy.add(w1, numpy.array(trainingSet[i])*int(trainingLabels[i]))
    return w1
def votedPerceptron(trainingSet, trainingLabels, w1):
    for i in range(len(trainingSet)):
        dotProduct = numpy.dot(numpy.array(trainingSet[i]), w1[-1][0])
        if float(trainingLabels[i] ) * float(dotProduct) <= 0:
            w1.append([numpy.add(w1[-1][0], numpy.array(trainingSet[i])*int(trainingLabels[i])), 1])
        else :
            w1[-1][1] += 1
    
    
def averagePerceptron(trainingSet, trainingLabels, w1, totalSum):
    for i in range(len(trainingSet)):
        dotProduct = numpy.dot(numpy.array(trainingSet[i]), w1)
        if float(trainingLabels[i] ) * float(dotProduct) <= 0:
            w1 = numpy.add(w1, numpy.array(trainingSet[i])*int(trainingLabels[i]))
            totalSum = numpy.add(totalSum, w1)
        else: 
            totalSum = numpy.add(totalSum, w1)
    return totalSum


def getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels):
    for i in range(len(trainingLabels)):
        label = trainingLabels[i]
        if int(label) == 1: 
            subsetLabels.append(1)
            subset.append(trainingSet[i])
        elif int(label) == 2:
            subsetLabels.append(-1)
            subset.append(trainingSet[i])

def getAccuracyRegular(testSet, testLabels, classifier):
    numWrong = 0
    for test in testSet:
        prediction  = numpy.dot(classifier, numpy.array(test))
        sign = numpy.sign(prediction)
        if sign == 0:
            numWrong += randint(0,1)
        if sign == testLabels[testSet.index(test)]:
            numWrong += 1
            
            

    return float(1) - float(numWrong)/float(len(testSet))

def getAccuracyVoted(testSet, testLabels, classifer):
    numWrong = 0

    for test in testSet:
        totalSum = 0
        for c in classifer:
            sign = numpy.sign(numpy.dot(c[0], numpy.array(test)))
            product = sign * c[1]
            totalSum += product
        actualPrediction = numpy.sign(totalSum)

        if actualPrediction == 0:
            numWrong += randint(0,1)
        if actualPrediction == testLabels[testSet.index(test)]:
            numWrong += 1
        
    return float(1) - float(numWrong)/float(len(testSet))

def getAccuracyAverage(testSet, testLabels, classifier):
    numWrong = 0
    for test in testSet:
        prediction  = numpy.dot(classifier, numpy.array(test))
        sign = numpy.sign(prediction)
        if sign == 0:
            numWrong += randint(0,1)
        if sign == testLabels[testSet.index(test)]:
            numWrong += 1
            
            

    return float(1) - float(numWrong)/float(len(testSet))
#main 
trainingSet = []
trainingLabels = []
testSet = []
testLabels= []
loadData('pa3train.txt', trainingSet, trainingLabels)
#loadData('practice.txt', trainingSet, trainingLabels)
loadData('pa3test.txt', testSet, testLabels)

subset = []
subsetLabels = []
testSubset = []
testSubsetLabels= []
getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels)
getLabelsOneAndTwo(testSubset, testSubsetLabels, testSet, testLabels)

dimension = 819
rounds = 2
#regular perceptron
finalW = []
for i in range (0, dimension):
    finalW.append(0)
for i in range(0, rounds):   
    finalW = regularPerceptron(subset, subsetLabels, finalW)
regularAccuracy = getAccuracyRegular(subset, subsetLabels, finalW)
print('regular error : ' + str(regularAccuracy))

#voted perceptron 
votedW1 = []
for i in range (0, dimension):
    votedW1.append(0)
votedOutput = []
votedOutput.append([votedW1,1])
for i in range(0, rounds):
    votedPerceptron(subset, subsetLabels, votedOutput)
votedAccuracy = getAccuracyVoted(subset, subsetLabels, votedOutput)
print('voted error '  + str(votedAccuracy))

#average perceptron 
averageW1 = []
average = []
for i in range (0, dimension):
    averageW1.append(0)
    average.append(0)
average = numpy.array(average)
for i in range(0, rounds):
    average = averagePerceptron(subset, subsetLabels, averageW1, average)
    averageAccuracy = getAccuracyAverage(subset, subsetLabels, average)
    print('average error ' + str(averageAccuracy))