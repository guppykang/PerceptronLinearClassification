import collections
import math
import numpy

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

def regularPerceptron(trainingSet, trainingLabels):
    #default w1
    w1 = []
    for i in range (0, 2):
        w1.append(0)

    for i in range(len(trainingSet)):
        dotProduct = numpy.dot(numpy.array(trainingSet[i]), w1)
        if float(trainingLabels[i] ) * float(dotProduct) <= 0:
            print('update')
            w1 = numpy.add(w1, numpy.array(trainingSet[i])*int(trainingLabels[i]))
        print(w1)

def getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels):
    for i in range(len(trainingLabels)):
        label = trainingLabels[i]
        if int(label) == 1: 
            subsetLabels.append(1)
            subset.append(trainingSet[i])
        elif int(label) == 2:
            subsetLabels.append(-1)
            subset.append(trainingSet[i])

#main 
trainingSet = []
trainingLabels = []
#loadData('pa3train.txt', trainingSet, trainingLabels)
loadData('practice.txt', trainingSet, trainingLabels)

subset = []
subsetLabels = []
getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels)

regularPerceptron(subset, subsetLabels)


