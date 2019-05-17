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

def loadWords(fileName, words=[]):
    fileOpen = open(fileName, "r")
    lines = fileOpen.readlines()
    fileOpen.close()
    for unFormattedLine in lines:
        line = unFormattedLine.split()
        currentLine = []
        for feature in line:
            words.append(str(feature))

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
    return [totalSum, w1]

def getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels):
     for i in range(len(trainingLabels)):
         label = trainingLabels[i]
         if int(label) == 1: 
             subsetLabels.append(1)
             subset.append(trainingSet[i])
         elif int(label) == 2:
             subsetLabels.append(-1)
             subset.append(trainingSet[i])

def separateForOneForAll(subset, subsetLabels, trainingSet, trainingLabels, one):
    count = 0
    for i in range(len(trainingLabels)):
        label = trainingLabels[i]
        if int(label) == int(one): 
            count += 1
            subsetLabels.append(1)
            subset.append(trainingSet[i])
        else :
            subsetLabels.append(-1)
            subset.append(trainingSet[i])
    return count

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

def findSixWords(words, w):
    copy = []
    for item in w: 
        copy.append(item)

    worstThree = []
    bestThree = []

    copy = sorted(copy)
    for i in copy[:3]:
        worstThree.append(words[w.tolist().index(i)])
    
    for i in copy[-3:]:
        bestThree.append(words[w.tolist().index(i)])

    bestThreeReverse = []
    for i in reversed(bestThree): 
        bestThreeReverse.append(i)
    return [worstThree, bestThreeReverse]

#main 
trainingSet = []
trainingLabels = []
testSet = []
testLabels= []
loadData('pa3train.txt', trainingSet, trainingLabels)
#if using practice, change dimension to 2
#loadData('practice.txt', trainingSet, trainingLabels)
loadData('pa3test.txt', testSet, testLabels)

subset = []
subsetLabels = []
testSubset = []
testSubsetLabels= []
getLabelsOneAndTwo(subset, subsetLabels, trainingSet, trainingLabels)
getLabelsOneAndTwo(testSubset, testSubsetLabels, testSet, testLabels)

words = []
loadWords('pa3dictionary.txt', words)


dimension = 819
rounds = 3
#regular perceptron
print('STARTING REGULAR PERCEPTRON ALGORITHM')
finalW = []
for i in range (0, dimension):
    finalW.append(0)
for i in range(0, rounds):   
    finalW = regularPerceptron(subset, subsetLabels, finalW)
    regularAccuracyTraining = getAccuracyRegular(subset, subsetLabels, finalW)
    regularAccuracyTesting = getAccuracyRegular(testSubset, testSubsetLabels, finalW)
    print('training regular error  for round ' + str(i+1) + "  " + str(regularAccuracyTraining))
    print('testing regular error  for round ' + str(i+1) + "  " + str(regularAccuracyTesting))

#voted perceptron 
print('STARTING VOTED PERCEPTRON ALGORITHM')
votedW1 = []
for i in range (0, dimension):
    votedW1.append(0)
votedOutput = []
votedOutput.append([votedW1,1])
for i in range(0, rounds):
    votedPerceptron(subset, subsetLabels, votedOutput)
    votedAccuracyTraining = getAccuracyVoted(subset, subsetLabels, votedOutput)
    votedAccuracyTesting = getAccuracyVoted(testSubset, testSubsetLabels, votedOutput)
    print('training voted error  for round ' + str(i+1) + "  "  + str(votedAccuracyTraining))
    print('testing voted error  for round ' + str(i+1) + "  "  + str(votedAccuracyTesting))

#average perceptron 
print('STARTING AVERAGE PERCEPTRON ALGORITHM')
averageW1 = []
average = []
for i in range (0, dimension):
    averageW1.append(0)
    average.append(0)
average = numpy.array(average)
for i in range(0, rounds):
    result = averagePerceptron(subset, subsetLabels, averageW1, average)
    averageW1 = result[1]
    average = result[0]
    if i == 2:
        threePasses = result[0]
    averageAccuracyTraining = getAccuracyAverage(subset, subsetLabels, result[0])
    averageAccuracyTesting = getAccuracyAverage(testSubset, testSubsetLabels, result[0])
    print('training average error for round ' + str(i+1) + "  " +  str(averageAccuracyTraining))
    print('testing average error for round ' + str(i+1) + "  " +  str(averageAccuracyTesting))


#part2
sixWords = findSixWords(words, threePasses)
print('smallest value' + str(sixWords[0]))
print('greatest value' + str(sixWords[1]))

#part3
#yer mam
onesSet = []
onesLabels = []
numOfOnes = separateForOneForAll(onesSet, onesLabels, trainingSet, trainingLabels, 1)
oneVsAllW = []
for i in range (0, dimension):
    oneVsAllW.append(0)
oneVsAll = regularPerceptron(onesSet, onesLabels, oneVsAllW)

twosSet = []
twosLabels = []
numOfTwos = separateForOneForAll(twosSet, twosLabels, trainingSet, trainingLabels, 2)
twoVsAllW = []
for i in range (0, dimension):
    twoVsAllW.append(0)
twoVsAll = regularPerceptron(twosSet, twosLabels, twoVsAllW)

threesSet = []
threesLabels = []
numOfThrees = separateForOneForAll(threesSet, threesLabels, trainingSet, trainingLabels, 3)
threeVsAllW = []
for i in range (0, dimension):
    threeVsAllW.append(0)
threeVsAll = regularPerceptron(threesSet, threesLabels, threeVsAllW)

foursSet = []
foursLabels = []
numOfFours = separateForOneForAll(foursSet, foursLabels, trainingSet, trainingLabels, 4)
fourVsAllW = []
for i in range (0, dimension):
    fourVsAllW.append(0)
fourVsAll = regularPerceptron(foursSet, foursLabels, fourVsAllW)

fivesSet = []
fivesLabels = []
numOfFives = separateForOneForAll(fivesSet, fivesLabels, trainingSet, trainingLabels, 5)
fiveVsAllW = []
for i in range (0, dimension):
    fiveVsAllW.append(0)
fiveVsAll = regularPerceptron(fivesSet, fivesLabels, fiveVsAllW)

sixsSet = []
sixsLabels = []
numOfSixs = separateForOneForAll(sixsSet, sixsLabels, trainingSet, trainingLabels, 6)
sixVsAllW = []
for i in range (0, dimension):
    sixVsAllW.append(0)
sixVsAll = regularPerceptron(sixsSet, sixsLabels, sixVsAllW)

confusionMatrix = [[0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0]]
for item in testSet:
    actualLabelIndex = int(testLabels[testSet.index(item)]) - 1
    predictions = []
    predictions.append(numpy.sign(numpy.dot(oneVsAll, numpy.array(item))))
    predictions.append(numpy.sign(numpy.dot(twoVsAll, numpy.array(item))))
    predictions.append(numpy.sign(numpy.dot(threeVsAll, numpy.array(item))))
    predictions.append(numpy.sign(numpy.dot(fourVsAll, numpy.array(item))))
    predictions.append(numpy.sign(numpy.dot(fiveVsAll, numpy.array(item))))
    predictions.append(numpy.sign(numpy.dot(sixVsAll, numpy.array(item))))

    moreThanOne = 0
    for i in predictions:
        if int(i) == 1:
            moreThanOne += 1
        if int(i) == 0: 
            moreThanOne += randint(0,1)


        if moreThanOne == 1:
            confusionMatrix[predictions.index(i)][actualLabelIndex] += 1
            break

    #prediction is don't know
    if moreThanOne == 0 :
        confusionMatrix[6][actualLabelIndex] += 1

sizes = [0,0,0,0,0,0]
for label in testLabels:
    sizes[int(label)-1] += 1
print(sizes)

for column in range(0,6):   
    for row in range(0, 7):
        confusionMatrix[row][column] = float(confusionMatrix[row][column])/float(sizes[column])

for row in confusionMatrix:
    print(row)


    



