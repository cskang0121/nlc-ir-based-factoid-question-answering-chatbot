# NLC : IR-Based Factoid Question Answering Chatbot

## Individual Project Completed By
- SMU Computer Science Year 4 – Kang Chin Shen (cskang.2020@scis.smu.edu.sg)

## Problem Statement

### Overview
This project endeavors to investigate various methodologies for the implementation of an **Information Retrieval (IR)-based Question Answering Chatbot**, as a crucial component of my research and studies in natural language communication. The high level view of the architecture of the information retrieval system is as follows:
![Screenshot 2023-09-14 at 9 42 33 PM](https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/43637109-25bc-4dbf-870d-a1c38f4e361d)

- The model performs indexing at the paragraph level rather than the document level. This allows the model to provide more fine-grained results based on user questions.
- The model calculates similarity scores between a query vector and stored data using three distinct mechanisms, namely:

&nbsp;&nbsp;**(1) Traditional TF-IDF computation**

![Screenshot 2023-09-14 at 10 40 36 PM](https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/04f33a8e-641a-424c-bfeb-15a97de8a6a3)

&nbsp;&nbsp;**(2) Consine Similarity**
![Screenshot 2023-09-14 at 10 41 38 PM](https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/84c2c9fe-4c18-44af-9530-3184d3175001)

&nbsp;&nbsp;**(3) BM25 Score Function**
![Screenshot 2023-09-14 at 10 42 49 PM](https://github.com/cskang0121/nlc-ir-based-factoid-question-answering-chatbot/assets/79074359/f9f8b3f0-9ac9-4162-9ceb-5a5af327ca96)


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
