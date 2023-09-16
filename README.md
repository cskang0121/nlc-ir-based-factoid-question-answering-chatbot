# NLC : IR-Based Factoid Question Answering Chatbot

## Individual Project Completed By
- SMU Computer Science Year 4 – Kang Chin Shen (cskang.2020@scis.smu.edu.sg)

## Motivation
NLP has emerged as one of the most prominent subjects in recent years, particularly following the debut of ChatGPT-3.5 in June 2022. This project endeavours to investigate various methodologies for the implementation of an **Information Retrieval (IR)-based Question Answering Chatbot**, as a crucial component of my research and studies in natural language communication. 

### Overview
The high level view of the architecture of the information retrieval system is as follows:
<img width="807" alt="Screenshot 2023-09-14 at 10 57 10 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/d82703ce-2553-48bc-9e30-04dc215907f0">

- The model performs indexing at the paragraph level rather than the document level. This allows the model to provide more fine-grained results based on user questions.
- The model calculates similarity scores between a query vector and stored data using three distinct mechanisms, namely

  (1) Traditional TF-IDF Model

  (2) Cosine Similarity Model

  (3) Okapi BM25 Model
- Note that the default setting is Okapi BM25 Model (as it outperformed the rest).
  
### Technologies & Datasets
&nbsp;&nbsp;[`Python 3.10`](https://www.python.org/downloads/)
&nbsp;&nbsp;[`NLTK 3.5`](https://www.nltk.org/install.html)
&nbsp;&nbsp;[`Stanford Question Answering Datasets`](https://rajpurkar.github.io/SQuAD-explorer/)

## Repository High Level Architecture
```
| nlc-ir-based-factoid-question-answering-chatbot       # Root folder

    | dataset                   # 12 topics downloaded from 'Standford Question Answering Datasets'

    StanfordDataset.py          # Loading of the downloaded datasets

    DocumentRetrievalModel.py   # Implementation of the core logic

    P2.py                       # Entry point of the chatbot

    testQA.py                   # Evaluation of the chatbot performance

    DataExtractor.py            # Tagging of temporal expressions in text

    ProcessedQuestion.py        # Query Formulation

    README.md                   # Code documentation

    Other files 
```
## Deep Dive Into Implementation

### (1) Traditional TF-IDF Model

<img width="840" alt="Screenshot 2023-09-15 at 9 36 41 AM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/cb09bf87-9469-4511-9580-57f0c20983a1">

&nbsp;&nbsp;**Terms' Explanation**:

- q: Query vector
- d: Paragraph vector (i.e., document vector in general – indexing is performed at paragraph level)
- t: Term
- tf(t,d): Term frequency of t in d
- idf(t): Inverted Paragraph Frequency (i.e., Inverted Document Frequency in general)
- |d|: Vector distance of paragraph vector
- count(t,d): Raw frequency of t in d
- df(t): Paragraph frequency of t (i.e., Document Frequency in general)

&nbsp;&nbsp;**Code Implementation**:

```python

# File: DocumentRetrievalModel.py 

def computeTFIDF(self):
    ...
    # Compute IDF
    self.idf = {}
    for word in wordParagraphFrequency:
        self.idf[word] = math.log((self.totalParas/wordParagraphFrequency[word]), 10)
    ...

def computeSimilarity(self, pInfo, queryVector, queryDistance):

    # Compute paragraph vector distance
    pVectorDistance = 0
    for word in pInfo['wF'].keys():
        pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
    pVectorDistance = math.pow(pVectorDistance,0.5)

    if(pVectorDistance == 0):
        return 0

    # Compute tf-idf score between query vector and paragraph vector
    total = 0
    for word in queryVector.keys():
        if word in pInfo['wF']:
            w = math.log(pInfo['wF'][word]+1, 10)
            idf = self.idf[word]
            total += w*idf
    sim = total / pVectorDistance

    return sim
```

&nbsp;&nbsp;**Model Evaluation**:

<img width="468" alt="Screenshot 2023-09-15 at 10 27 53 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/8d5cf987-78a1-4f39-8a55-71df901ab759">

- Averaged Overall Accuracy: 69.706%

### (2) Cosine Similarity Model

<img width="854" alt="Screenshot 2023-09-15 at 10 12 03 AM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/8298690b-0865-4894-9565-78e0623d586f">

&nbsp;&nbsp;**Terms' Explanation**:

- q: Query vector
- d: Paragraph vector (i.e., document vector in general – indexing is performed at paragraph level)
- t: Term
- tf(t,q): Term frequency of t in q
  - Note that tf(t,q) is raw frequency without applying any logarithm function
- tf(t,d): Term frequency of t in d
  - Note that tf(t,d) is raw frequency without applying any logarithm function
- idf(t): Inverted Paragraph Frequency (i.e., Inverted Document Frequency in general)
  - Note that Laplace Smoothing technique is applied here
- |q|: Vector distance of query vector
- |d|: Vector distance of paragraph vector
- df(t): Paragraph frequency of t (i.e., Document Frequency in general)

&nbsp;&nbsp;**Code Implementation**:

```python

# File: DocumentRetrievalModel.py 

def computeTFIDF(self):
    ...
    # Compute IDF with Laplace Smoothing
    self.idf = {}
    for word in wordParagraphFrequency:
        self.idf[word] = math.log((self.totalParas+1)/wordParagraphFrequency[word]) 
    ...

def computeSimilarity(self, pInfo, queryVector, queryDistance):

    # Compute paragraph vector distance
    pVectorDistance = 0
    for word in pInfo['wF'].keys():
        pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
    pVectorDistance = math.pow(pVectorDistance,0.5)

    if(pVectorDistance == 0):
        return 0

    # Compute cosine similar between query vector and paragraph vector
    dotProduct = 0
    for word in queryVector.keys():
        if word in pInfo['wF']:
            q = queryVector[word]
            w = pInfo['wF'][word]
            idf = self.idf[word]
            dotProduct += q*w*idf*idf
    sim = dotProduct / (pVectorDistance * queryDistance)

    return sim
```

&nbsp;&nbsp;**Model Evaluation**:

<img width="469" alt="Screenshot 2023-09-15 at 10 29 01 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/bc121a62-5364-4f30-888a-9010699a5caa">

- Averaged Overall Accuracy: 70.167%

### (3) Okapi BM25 Model

![Screenshot 2023-09-16 at 12 31 54 PM](https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/51c3e625-d275-41ed-89fd-15193154e188)

&nbsp;&nbsp;**Terms' Explanation**:

- q: Query vector
- d: Paragraph vector (i.e., document vector in general – indexing is performed at paragraph level)
- t: Term
- N: Total number of paragraphs
- df(t): Paragraph frequency of t (i.e., Document Frequency in general)
- tf(t,d): Term frequency of t in d
  - Note that tf(t,d) is raw frequency without applying any logarithm function
- k: A knob to adjust the balance between TF and IDF
  - k = 1.2 is used in this project
- b: A value to control the importance of document length normalisation
  - b = 0.75 is used is this project 
- |d|: Vector distance of paragraph vector
  - Note that vector distance is absolute number of words in d
- |d(avg)|: Average vector distance of paragraph vector
- count(t,d): Raw frequency of t in d

&nbsp;&nbsp;**Code Implementation**:
```python
def __init__(self,paragraphs,removeStopWord = False,useStemmer = False):
    ...
    # Set the required hyperparameters for BM25 function 
    self.k = 1.2
    self.b = 0.75
    self.avgParaLength = self.computeAvgParaLength()

def computeTFIDF(self):
    ...
    # Compute IDF
    for word in wordParagraphFrequency:
        self.idf[word] = math.log((self.totalParas/wordParagraphFrequency[word]), 10)
    ...

def getSimilarParagraph(self,queryVector):
    # Rank paragraphs based on computeBM25Similarity()
    pRanking = []
    for index in range(0,len(self.paragraphInfo)):
        sim = self.computeBM25Similarity(self.paragraphInfo[index], queryVector, self.avgParaLength, self.idf)
        pRanking.append((index,sim))
    return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]

def getMostRelevantSentences(self, sentences, pQ, nGram=3):
    # Compute details at sentence level
    relevantSentences = []

    totalSents = len(sentences)

    sentenceInfo = {}
    for index in range(0,len(sentences)):
        wordFrequency = self.getTermFrequencyCount(sentences[index])
        sentenceInfo[index] = {}
        sentenceInfo[index]['wF'] = wordFrequency

    wordSentenceFrequency = {}
    for index in range(0,len(sentences)):
        for word in sentenceInfo[index]['wF'].keys():
            if word in wordSentenceFrequency.keys():
                wordSentenceFrequency[word] += 1
            else:
                wordSentenceFrequency[word] = 1

    sentIdf = {} 
    for word in wordSentenceFrequency:
        sentIdf[word] = math.log((totalSents/wordSentenceFrequency[word]), 10)

    avgSentLength = self.computeAvgSentLength(sentenceInfo, totalSents)

    # Rank sentences based on computeBM25Similarity()
    for index in range(0,len(sentenceInfo)):
        sim = self.computeBM25Similarity(sentenceInfo[index], pQ.qVector, avgSentLength, sentIdf)
        relevantSentences.append((sentences[index],sim))
    
    return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)

# Compute BM25 score between query vector and paragraph vector
def computeBM25Similarity(self, context, queryVector, avgVectorLength, contextIdf):
    vectorDistance = sum(context['wF'].values())
    
    sim = 0
    for word in queryVector.keys():
        if word in context['wF']:

            # IDF
            idf = contextIdf[word]
            
            # Weighted TF
            w = math.log(context['wF'][word]+1, 10)
            w = w / (self.k * (1 - self.b + (self.b * vectorDistance / avgVectorLength)) + w)

            sim += idf * w
    
    return sim

# Compute average paragraph length
def computeAvgParaLength(self):
    totalWordCounts = 0
    for index in self.paragraphInfo.keys():
        totalWordCounts += sum(self.paragraphInfo[index]['wF'].values())
    avgParaLength = totalWordCounts / self.totalParas

    return avgParaLength

# Compute average sentence length
def computeAvgSentLength(self, sentenceInfo, totalSents):
    totalWordCounts = 0
    for index in sentenceInfo.keys():
        totalWordCounts += sum(sentenceInfo[index]['wF'].values())
    avgSentLength = totalWordCounts / totalSents

    return avgSentLength
```
&nbsp;&nbsp;**Model Evaluation**:

<img width="465" alt="Screenshot 2023-09-16 at 3 30 48 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/2052d050-e233-41a9-bb5e-0ca3191ad6e9">

- Averaged Overall Accuracy: 75.640%

## Running The Code
1. Run the command ```git clone https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot.git``` in a new terminal on your local machine.
2. ```cd``` to ```nlc-ir-based-factoid-question-answering-chatbot``` folder.

3.1. To evaluate the model: Run ```python testQA.py```

3.2. To interact with the model: Run ```python P2.py dataset/<dataset>.txt```
   - Note that dataset can be chosen from ```/dataset``` folder, e.g., ```python P2.py dataset/Marvel_Comics.txt```

## Credits
> Special thanks to **Vaibhaw Raj** for providing the base implementation of this project!

## References
- [Speech and Language Processing. Daniel Jurafsky & James H. Martin.](http://web.stanford.edu/~jurafsky/slp3/14.pdf)
- [Base Implementation of Cosine Similarity Approach](https://github.com/vaibhawraj)
