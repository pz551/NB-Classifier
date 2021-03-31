#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 00:17:32 2019

@author: puzhao
"""

import csv

import math

#Handling unknown words in test corpus
unknown = "*UNK*"

#Read data from training set
def readData(fileName):
	data = []
	file = open(fileName, "r")

	for word in file.read().split():
		data.append(word)

	file.close()
	return data

#Read data from test set
def readTestFile(fileName):
    reviews = []
    file = open(fileName, "r")
    lines = file.readlines()
    
    for line in lines:
        reviews.append(line)
        
    file.close()
    return reviews

#Create unigram and bigram counts for the training data
def createBigram(data):
    listOfBigrams = []
    bigramCounts = {}
    unigramCounts = {}

    for i in range(len(data)):
        if i < len(data) - 1:

            if (data[i], data[i+1]) in bigramCounts:
                bigramCounts[(data[i], data[i + 1])] += 1
                listOfBigrams.append((data[i], data[i + 1]))
            else:
                bigramCounts[(data[i], data[i + 1])] = 0
                listOfBigrams.append((unknown, unknown))

        if data[i] in unigramCounts:
            unigramCounts[data[i]] += 1
        else:
            unigramCounts[data[i]] = 0
    
    #handling unknown words by replacing the first occurence of each unigram/bigram with <UNK>
    unigramSize = len(unigramCounts)
    unigramCounts[unknown] = unigramSize
    bigramSize = len(bigramCounts)
    bigramCounts[(unknown, unknown)] = bigramSize
    
    return listOfBigrams, unigramCounts, bigramCounts

#Calculate the probability of N-gram model without smoothing
def calcProb(listOfBigrams, unigramCounts, bigramCounts):
    listOfUnigramProb={}
    for unigram in unigramCounts:
        listOfUnigramProb[unigram] = (unigramCounts[unigram]) / (len(unigramCounts))
        
    listOfBigramProb = {}
    for bigram in listOfBigrams:
        word1 = bigram[0]
		
        listOfBigramProb[bigram] = (bigramCounts.get(bigram))/(unigramCounts.get(word1))

    return listOfUnigramProb, listOfBigramProb

#Add-k smoothing
def additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, k):
	listOfProb = {}

	for bigram in listOfBigrams:
		word1 = bigram[0]
		listOfProb[bigram] = (bigramCounts[bigram] + k)/(unigramCounts[word1] + len(unigramCounts) * k)

	return listOfProb

#Perplexity function
def perplexity(listOfProb, review):
    data = review.split()
    power = 0
    
    for i in range(len(data) - 1):
        try:
            prob = listOfProb[(data[i], data[i + 1])]
        except KeyError:
            prob = listOfProb[(unknown, unknown)]
        power -= math.log(prob)
        
    power = power / len(data)
    
    return math.pow(math.e, power)

#Tuning hyperparameter k using the validation set
def validateTruth(truthfulModel, deceptiveModel, validationSet):
    correct = 0
    for review in validationSet:
        truthfulPerp = perplexity(truthfulModel, review)
        deceptivePerp = perplexity(deceptiveModel, review)
        if truthfulPerp <= deceptivePerp:
            correct += 1
    return (correct / len(validationSet))

def validateDeception(truthfulModel, deceptiveModel, validationSet):
    correct = 0
    for review in validationSet:
        truthfulPerp = perplexity(truthfulModel, review)
        deceptivePerp = perplexity(deceptiveModel, review)
        if deceptivePerp <= truthfulPerp:
            correct += 1
    return (correct / len(validationSet))
        
def tuningHyperparameter(start,end):
    with open("TuningHyper.csv", "w") as file:
        fileWriter = csv.writer(file, delimiter=",")
        fileWriter.writerow(["Hyperparameter", "Acc Truthful", "Acc Deceptive", "Average Accurancy"])
        for i in range(start,end+1):


            #Constructing the bigram model for truthful corpus in i-smoothing
            truthful = readData("train/truthful.txt")
            listOfBigrams, unigramCounts, bigramCounts = createBigram(truthful)
            truthtulUnigram, truthfulBigram = calcProb(listOfBigrams, unigramCounts, bigramCounts)
            truthfulModel = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, i)

            #Constructing the bigram model for deceptive corpus in i-smoothing
            deceptive = readData("train/deceptive.txt")
            listOfBigrams, unigramCounts, bigramCounts = createBigram(deceptive)
            deceptiveUnigram, deceptiveBigram = calcProb(listOfBigrams, unigramCounts, bigramCounts)
            deceptiveModel = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, i)

            #Classify the test corpus with existing Language Models
            truthfulVal = readTestFile("validation/truthful.txt")
            deceptiveVal = readTestFile("validation/deceptive.txt")
            accOnTruth = 1-validateTruth(truthfulModel, deceptiveModel, truthfulVal)
            accOnDeception = 1-validateDeception(truthfulModel, deceptiveModel, deceptiveVal)
            fileWriter.writerow([i, str(accOnTruth), str(accOnDeception), str((accOnTruth+accOnDeception)/2)])
        
#Classifier using language models
def classify(k):
    #Reading the test file
    test = readTestFile("test/test.txt")

    #Constructing the bigram model for truthful corpus
    truthful = readData("train/truthful.txt")
    listOfBigrams, unigramCounts, bigramCounts = createBigram(truthful)
    truthtulUnigram, truthfulBigram = calcProb(listOfBigrams, unigramCounts, bigramCounts)
    truthfulModel = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, k)

    #Constructing the bigram model for deceptive corpus
    deceptive = readData("train/deceptive.txt")
    listOfBigrams, unigramCounts, bigramCounts = createBigram(deceptive)
    deceptiveUnigram, deceptiveBigram = calcProb(listOfBigrams, unigramCounts, bigramCounts)
    deceptiveModel = additiveSmoothing(listOfBigrams, unigramCounts, bigramCounts, k)
    
    with open("Prediction.csv", "w") as file:
        fileWriter = csv.writer(file, delimiter=",")
        fileWriter.writerow(["Id", "Prediction"])
        index = 0
        for review in test:
            truthfulPerp = perplexity(truthfulModel, review)
            deceptivePerp = perplexity(deceptiveModel, review)
            if truthfulPerp <= deceptivePerp:
                fileWriter.writerow([index, 1])
            else:
                fileWriter.writerow([index, 0])
            index += 1
    return 0
   
tuningHyperparameter(1,100)
         
classify(9) #k=9..15