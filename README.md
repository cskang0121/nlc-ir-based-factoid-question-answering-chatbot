# NLC : IR-Based Factoid Question Answering Chatbot

## Individual Project Completed By
- SMU Computer Science Year 4 – Kang Chin Shen (cskang.2020@scis.smu.edu.sg)

## Problem Statement

### Overview
This project endeavors to investigate various methodologies for the implementation of an **Information Retrieval (IR)-based Question Answering Chatbot**, as a crucial component of my research and studies in natural language communication. The high level view of the architecture of the information retrieval system is as follows:
<img width="807" alt="Screenshot 2023-09-14 at 10 57 10 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/d82703ce-2553-48bc-9e30-04dc215907f0">

- The model performs indexing at the paragraph level rather than the document level. This allows the model to provide more fine-grained results based on user questions.
- The model calculates similarity scores between a query vector and stored data using three distinct mechanisms, namely:

&nbsp;&nbsp;**(1) Traditional TF-IDF**
<img width="840" alt="Screenshot 2023-09-15 at 9 36 41 AM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/cb09bf87-9469-4511-9580-57f0c20983a1">

&nbsp;&nbsp;**Terms' Explanatioin**:

- q: Query vector
- d: Paragraph vector (i.e., document vector in general – indexing is performed at paragraph level)
- t: Term
- tf(t,d): Term frequency of t in d
- idf(t): Inverted Paragraph Frequency (i.e., Inverted Document Frequency in general)
- |d|: Vector distance of paragraph vector
- count(t,d): Raw frequency of t in d
- df(t): Paragraph frequency of t (i.e., Document Frequency in general)

&nbsp;&nbsp;**(2) Consine Similarity**
<img width="854" alt="Screenshot 2023-09-15 at 10 12 03 AM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/8298690b-0865-4894-9565-78e0623d586f">

&nbsp;&nbsp;**Terms' Explanatioin**:

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

&nbsp;&nbsp;**(3) Okapi BM25**
<img width="818" alt="Screenshot 2023-09-15 at 10 12 22 AM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/d24e178f-1421-4096-8b4b-18e8deea082a">

&nbsp;&nbsp;**Terms' Explanatioin**:

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
- idf(t): Inverted Paragraph Frequency (i.e., Inverted Document Frequency in general)

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
## Model Design

## Running The Code

## Credits
> Special thanks to **Vaibhaw Raj** for providing the base implementation of this project!

## References
- [Speech and Language Processing. Daniel Jurafsky & James H. Martin.](http://web.stanford.edu/~jurafsky/slp3/14.pdf)
- [Base Implementation of Cosine Similarity Approach](https://github.com/vaibhawraj)
