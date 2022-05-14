Incremental QA Helps Passage Retrieval via Curriculum Learning
=

Prepare the dataset:
------------
- Using gather_resources.sh to download the QANTA dataset, SQuAD dataset, and model weights

Problem Statement
------------------
The goal is to build **A document retriever** that reads an open-domain question text and finds the page from a set of documents that is relevant and contains the answer. 

The retriever has two sequential steps:
  -  1. Predict some top K wikipedia page-id guesses using a TF-IDF (or better) guesser.
  -  2. Rerank the top guesses using a BERT based **Reranker** and output the best page.

For the reranker part, we train the model using Curriculum Learning.

The performance of the system is measured using top-k metrics, i.e., the rate at which the top-k passages will contain the correct passage.

File Descriptions:
-

* `base_models.py`: Contains the abstract base classes that provides an idea of High Level API for each component. 

* `models.py`: The model implementation for out extractor.

* `tfidf_guesser.py`: TF-IDF based Guesser. 

* `qbdata.py`: Class files and util methods for QuizBowl questions and dataset.

* `models/` the directory for reranker training w/o Curriculum Learning
* `new_models/` the directory for reranker training w/ Curriculum Learning
* `train_new_guesser.py` the file for reranker training w/o Curriculum Learning
* `train_new_guesser_ct.py` the file for reranker training w/ Curriculum Learning
* `test_guesser.py` the file for reranker testing w/o Curriculum Learning
* `test_guesser_ct.py` the file for reranker testing w/ Curriculum Learning
* `parse_squad_normal.py` the file for reranker testing w/o Curriculum Learning on SQuAD
* `parse_squad_ct.py` the file for reranker testing w/ Curriculum Learning on SQuAD

System Environment
-------------------
The package version are provided in the `requirements.txt` file. We also use WANDB for model training.
```
transformers==4.16.2
sentence-transformers==2.2.0
datasets==1.18.3
torch==1.9.0
torchvision==0.10.0
torchaudio==0.9.0
scikit-learn==0.24.2
nltk==3.6.7
spacy==3.2.2
wandb
```

