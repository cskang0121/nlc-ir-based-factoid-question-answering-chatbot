# NLC : IR-Based Factoid Question Answering Chatbot

## Individual Project Completed By
- SMU Computer Science Year 4 – Kang Chin Shen (cskang.2020@scis.smu.edu.sg)

## Problem Statement

### Overview
This project endeavors to investigate various methodologies for the implementation of an **Information Retrieval (IR)-based Question Answering Chatbot**, as a crucial component of my research and studies in natural language communication. The high level view of the architecture of the information retrieval system is as follows:
<img width="807" alt="Screenshot 2023-09-14 at 10 57 10 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/d82703ce-2553-48bc-9e30-04dc215907f0">

- The model performs indexing at the paragraph level rather than the document level. This allows the model to provide more fine-grained results based on user questions.
- The model calculates similarity scores between a query vector and stored data using three distinct mechanisms, namely:

&nbsp;&nbsp;**(1) Traditional TF-IDF computation**

<img width="840" alt="Screenshot 2023-09-14 at 10 58 20 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/84f38ebd-e425-4602-83a4-b10cd1478c1c">

&nbsp;&nbsp;**(2) Consine Similarity**

<img width="859" alt="Screenshot 2023-09-14 at 10 58 57 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/02e1a8d6-4153-40ab-bfe7-33aeae053933">

&nbsp;&nbsp;**(3) BM25 Score Function**

<img width="835" alt="Screenshot 2023-09-14 at 10 59 44 PM" src="https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/391a7627-710d-436d-adb2-09b769d3162e">

### Technologies & Datasets
&nbsp;&nbsp;[`Python 3.10`](https://www.python.org/downloads/)
&nbsp;&nbsp;[`NLTK 3.5`](https://www.nltk.org/install.html)
&nbsp;&nbsp;[`Stanford Question Answering Datasets`](https://rajpurkar.github.io/SQuAD-explorer/)

## Repository High Level Architecture

## Model Design

## Running The Code

## Credits
> Special thanks to **Vaibhaw Raj** for providing the base implementation of this project!

## References
- [Speech and Language Processing. Daniel Jurafsky & James H. Martin.](http://web.stanford.edu/~jurafsky/slp3/14.pdf)
- [Base Implementation of Cosine Similarity Approach](https://github.com/vaibhawraj)
