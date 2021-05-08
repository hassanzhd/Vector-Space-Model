import os
root = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(root, 'nltk_data')

import nltk
nltk.data.path.append(download_dir)

import re
from math import log, sqrt


class VectorSpaceModel: # helper class implementing the model
  def __init__(self):
    self.totalNumberOfDocuments = 50
    self.index = {}
    self.stemmer = nltk.stem.porter.PorterStemmer()
    self.indexFileName = "./static/index.txt"
    self.stopWordFileName = "./Stopword-List.txt"
    self.documentsPath = "./static/ShortStories/{}.txt"
    self.stopWords = self.getStopWordsFromFile()
    self.readIndexOrPreprocess()

  """ returns stop words  """
  def getStopWordsFromFile(self): 
    fp = open(self.stopWordFileName,'r')
    stopWords = fp.read()
    stopWords = nltk.word_tokenize(stopWords) # converting text to list of tokens
    fp.close()
    return (stopWords)

  """ returns all normalized tokens of document """
  def getTokensFromFile(self, __fileName):
    punctuationRegex = "[.,!?:;‘’”“\"]"
    fp = open(__fileName, 'r')
    text = fp.read().lower()  # case folding
    text = re.sub(punctuationRegex, "", text) # removal of punctuatiom
    text = re.sub("[-]", " ", text)
    tokens = nltk.word_tokenize(text) # converting text to list of tokens
    fp.close()
    return (tokens)

  """ creates inverted index containing all terms of documents """
  def createIndex(self):
    currentDocument = 1

    while(currentDocument<=self.totalNumberOfDocuments):
      fileTokens = self.getTokensFromFile(self.documentsPath.format(currentDocument))

      for word in fileTokens:
        if (not (word in self.stopWords)):
          if(not (word in self.index)):
            self.index[word] = {'termFrequencies' : [0]*self.totalNumberOfDocuments, 'documentFrequency' : 0, 'idf': 0, 'tf-id-frequencies' : [0]*self.totalNumberOfDocuments }
            self.index[word]['termFrequencies'][currentDocument - 1] += 1
            self.index[word]['documentFrequency'] += 1
          else:
            if(self.index[word]['termFrequencies'][currentDocument - 1] >= 1):
              self.index[word]['termFrequencies'][currentDocument - 1] += 1
            else:
              self.index[word]['termFrequencies'][currentDocument - 1] += 1
              self.index[word]['documentFrequency'] += 1
    
      currentDocument += 1

    for word in self.index:
      self.index[word]['idf'] = log(self.totalNumberOfDocuments / self.index[word]['documentFrequency'], 10)

    for word in self.index:
      for i in range(self.totalNumberOfDocuments):
        self.index[word]['tf-id-frequencies'][i] = self.index[word]['termFrequencies'][i] * self.index[word]['idf']

  """ writes inverted index to index.txt """
  def writeInvertedIndex(self):
    fp = open(self.indexFileName, 'w')
    fp.write(str(self.index))
    fp.close()

  """ reads invertedIndex if index.txt exists otherwise creates and writes invertedIndex"""
  def readIndexOrPreprocess(self):
    try:
      fp = open(self.indexFileName, 'r')
      data = fp.read()
      self.index = eval(data)
      fp.close()
    except FileNotFoundError:
      self.createIndex()
      self.writeInvertedIndex()

  def createQueryVector(self, __queryTerms):
    queryVector = {}

    for word in __queryTerms:
      if (not (word in self.stopWords) and (word in self.index)):
        if(not (word in queryVector)):
          queryVector[word] = {'termFrequency' : 0, 'tf-id-frequency' : 0 }
          queryVector[word]['termFrequency'] += 1
        else:
            queryVector[word]['termFrequency'] += 1

    for word in queryVector:
      queryVector[word]['tf-id-frequency'] = queryVector[word]['termFrequency'] * self.index[word]['idf']

    return (queryVector)

  def dotProduct(self, __queryVector, __document):
    sumOfProducts = 0

    for word in __queryVector:
      sumOfProducts += (__queryVector[word]['tf-id-frequency'] * self.index[word]['tf-id-frequencies'][__document])

    return (sumOfProducts)

  def magnitudeProduct(self, __queryVector, __document):
    queryMagnitude = 0
    documentMagnitude = 0

    for word in __queryVector:
      queryMagnitude += pow(__queryVector[word]['tf-id-frequency'], 2)
      documentMagnitude += pow(self.index[word]['tf-id-frequencies'][__document], 2)

    queryMagnitude = sqrt(queryMagnitude)
    documentMagnitude = sqrt(documentMagnitude)

    return (queryMagnitude * documentMagnitude)

  def cosineSimilarity(self, __queryVector):
    similarityValues = []

    for i in range(self.totalNumberOfDocuments):
      try:
        similarityValues.append((self.dotProduct(__queryVector, i) / self.magnitudeProduct(__queryVector, i), i + 1))
      except:
        pass

    similarityValues.sort(reverse=True)

    return (similarityValues)

  def filterDocuments(self, __documents, __alphaValue):
    filteredDocuments = []

    for document in __documents:
      if document[0] >= __alphaValue:
        filteredDocuments.append(document)

    return (filteredDocuments)

  def executeQuery(self, __query):
    parsedQuery = nltk.word_tokenize(__query)
    queryVector = self.createQueryVector(parsedQuery[0:-1])
    documents = self.cosineSimilarity(queryVector)
    filteredDocuments = self.filterDocuments(documents, float(parsedQuery[-1]))
    return (filteredDocuments)
