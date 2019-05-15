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


#main 
trainingSet = []
trainingLabels = []
#loadData('pa3train.txt', trainingSet, trainingLabels)
loadData('practice.txt', trainingSet, trainingLabels)

#default w1
w1 = []
for i in range (0, 2):
    w1.append(0)

#store all the classifiers
classifiers = []
classifiers.append(w1)

for i in range(len(trainingSet)):
    dotProduct = numpy.dot(numpy.array(trainingSet[i]), classifiers[i])
    if float(trainingLabels[i] ) * float(dotProduct) <= 0:
        print('update')
        newClassifier = numpy.add(classifiers[-1], numpy.array(trainingSet[i])*int(trainingLabels[i]))
    else: 
        newClassifier = classifiers[-1]
    print(newClassifier)
    classifiers.append(newClassifier)

