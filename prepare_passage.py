from typing import List, Union
import numpy as np
import pandas as pd
from base_models import BaseGuesser, BaseReRanker, BaseRetriever, BaseAnswerExtractor
from qbdata import WikiLookup, QantaDatabase, Question
from typing import Optional, List, Tuple
from os.path import exists
from tqdm import tqdm
import pickle
import torch
from transformers import BertForSequenceClassification, pipeline
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification, \
    PreTrainedTokenizer
from transformers import TrainingArguments, Trainer
from tfidf_guesser import TfidfGuesser
from datasets import load_dataset, Dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, util

wiki_lookup = WikiLookup('../custom_data/wiki_lookup.json')
train_questions = QantaDatabase('../data/qanta.train.2018.json').train_questions
page_set = set()
for tq in train_questions:
    page = tq.page
    page_set.add(page)
passage_texts = []
answers = []
for page in page_set:
    answers.append(page)
    passage_texts.append(wiki_lookup[page]['text'])
print(len(answers))
filepath = 'models/passage.pickle'
with open(filepath, 'wb') as f:
    pickle.dump({
        'passage_texts': passage_texts,
        'answers': answers,
    }, f)
