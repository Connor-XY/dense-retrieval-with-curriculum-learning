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

# Change this based on the GPU you use on your machine
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def make_data_dict(examples: List[Question]):
    data_dict = {"text": [], "page": [], "first_sentence": [], "last_sentence": [], "tokenization": [], "answer": []}
    for question in examples:
        data_dict["text"].append(question.text)
        data_dict["page"].append(question.page)
        data_dict["first_sentence"].append(question.first_sentence)
        data_dict["last_sentence"].append(question.sentences[-1])
        data_dict["tokenization"].append(question.tokenizations)
        data_dict["answer"].append(question.answer)
    return data_dict


def generate_gold_answers(answer_text: str):
    answer_text = answer_text.lower()
    if '[' in answer_text:
        answers = set()
        main_answer, remaining = answer_text.split('[', 1)
        answers.add(main_answer.strip())
        phrases = remaining.split(';')
        for p in phrases:
            p = p.strip()
            if not p.startswith('or '):
                continue

            p = p[3:]
            i, j = 0, len(p) - 1

            while i <= j and p[i] in {'"', '\'', '[', ' '}:
                i += 1

            while i <= j and p[i] in {'"', '\'', ']', ' '}:
                j -= 1
            answers.add(p[i:j + 1])
        return answers
    return {answer_text.strip()}


class Guesser(BaseGuesser):
    """You can implement your own Bert based Guesser here"""

    def __init__(self):
        """
        Initializes data structures that will be useful later.
        """
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1').to(device)
        self.embedding_matrix = None
        self.i_to_ans = None

    def train(self, training_data: QantaDatabase, limit: int = -1) -> None:
        """
        Keyword arguments:
        training_data -- The dataset to build representation from
        limit -- How many training data to use (default -1 uses all data)
        """

        questions = [x.text for x in training_data.train_questions]
        answers = [x.page for x in training_data.train_questions]

        if limit > 0:
            questions = questions[:limit]
            answers = answers[:limit]

        x_array = []
        y_array = []

        for doc, ans in zip(questions, answers):
            x_array.append(doc)
            y_array.append(ans)
        self.i_to_ans = {i: ans for i, ans in enumerate(y_array)}
        self.embedding_matrix = self.model.encode(x_array)

    def guess(self, questions: List[str], max_n_guesses: Optional[int]) -> List[List[Tuple[str, float]]]:
        representations = self.model.encode(questions)
        guess_matrix = self.embedding_matrix.dot(representations.T).T
        guess_indices = (-guess_matrix).argsort(axis=1)[:, 0:max_n_guesses]
        guesses = []
        for i in range(len(questions)):
            idxs = guess_indices[i]
            guesses.append([(self.i_to_ans[j], guess_matrix[i, j]) for j in idxs])

        return guesses

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                'i_to_ans': self.i_to_ans,
                'embedding_matrix': self.embedding_matrix
            }, f)

    def load(self, filepath):
        """
        Load the guesser from a saved file
        """

        with open(filepath, 'rb') as f:
            params = pickle.load(f)
            self.embedding_matrix = params['embedding_matrix']
            self.i_to_ans = params['i_to_ans']


class NewGuesser(BaseGuesser):

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.guesser = TfidfGuesser()
        self.guesser.load('models/tfidf.pickle')
        self.wiki_lookup = WikiLookup('../custom_data/wiki_lookup.json')

    def make_whole_dataset(self, examples, str):
        data_dict = {"text": [], "page": [], "first_sentence": [], "last_sentence": [], "tokenization": [],
                     "answer": []}
        for question in examples:
            data_dict["text"].append(question.text)
            data_dict["page"].append(question.page)
            data_dict["first_sentence"].append(question.first_sentence)
            data_dict["last_sentence"].append(question.sentences[-1])
            data_dict["tokenization"].append(question.tokenizations)
            data_dict["answer"].append(question.answer)

        questions = data_dict['text']  # list of text
        answers = data_dict['page']  # list of answer
        tokenization = data_dict['tokenization']  # list of tokenization
        ref_texts = []  # list of wiki passage
        for page in answers:
            ref_texts.append(self.wiki_lookup[page]['text'])

        l = len(questions)  # num of questions
        n = 2  # num of negative samples
        # max_length = 512
        # passage_len = 250     # max len of passage
        k = 1000  # process k at a time

        sentences = []  # 'sentence'
        passages = []  # 'passage'
        gold_pages = []  # 'gold_page'
        passage_pages = []  # 'passage_page'
        labels = []  # 'label'

        print('start processing dataset')

        for i in tqdm(range((l + k - 1) // k)):
            text = questions[i * k: (i + 1) * k]  # group size
            token = tokenization[i * k: (i + 1) * k]  # group size
            answer = answers[i * k: (i + 1) * k]  # group size
            ref_text = ref_texts[i * k: (i + 1) * k]  # group size

            group_size = len(answer)
            for j in range(group_size):
                list_of_sentences = [text[j][start:end] for start, end in token[j]]
                page = answer[j]
                passage = ref_text[j]
                for sentence in list_of_sentences:
                    sentences.append(sentence)
                    passages.append(passage)
                    gold_pages.append(page)
                    passage_pages.append(page)
                    labels.append(1)

            list_text = text
            part_guesses = self.guesser.guess_wrong(list_text, n)
            for z in range(group_size):
                list_of_sentences = [text[z][start:end] for start, end in token[z]]
                part_guess = part_guesses[z]
                page = answer[z]
                cnt = 0
                for j in range(n):
                    neg_passage_page = part_guess[j][0]
                    if neg_passage_page == page:
                        continue
                    cnt += 1
                    neg_passage = self.wiki_lookup[neg_passage_page]['text']
                    for sentence in list_of_sentences:
                        sentences.append(sentence)
                        passages.append(neg_passage)
                        gold_pages.append(page)
                        passage_pages.append(neg_passage_page)
                        labels.append(0)
                    if cnt == n - 1:
                        break

        print('end processing dataset')
        print('total input size: {}'.format(len(labels)))
        new_data_dict = {'sentence': sentences, 'passage': passages, 'gold_page': gold_pages,
                         'passage_page': passage_pages, 'label': labels}
        answer_set = set()
        for answer in answers:
            answer_set.add(answer)
        print('total num of answers: {}'.format(len(answer_set)))
        pd_frame = pd.DataFrame.from_dict(new_data_dict)
        if str == "train":
            pd_frame.to_parquet('models/train.parquet')
        else:
            pd_frame.to_parquet('models/eval.parquet')

    def make_whole_data_dict(self, data_dict, str):
        questions = data_dict['text']  # list of text
        answers = data_dict['page']  # list of answer
        tokenization = data_dict['tokenization']  # list of tokenization
        ref_texts = []  # list of wiki passage
        for page in answers:
            ref_texts.append(self.wiki_lookup[page]['text'])

        l = len(questions)  # num of questions
        n = 2  # num of negative samples
        # max_length = 512
        # passage_len = 250     # max len of passage
        k = 1000  # process k at a time

        sentences = []  # 'sentence'
        passages = []  # 'passage'
        gold_pages = []  # 'gold_page'
        passage_pages = []  # 'passage_page'
        labels = []  # 'label'

        print('start processing dataset')

        for i in tqdm(range((l + k - 1) // k)):
            text = questions[i * k: (i + 1) * k]  # group size
            token = tokenization[i * k: (i + 1) * k]  # group size
            answer = answers[i * k: (i + 1) * k]  # group size
            ref_text = ref_texts[i * k: (i + 1) * k]  # group size

            group_size = len(answer)
            for j in range(group_size):
                list_of_sentences = [text[j][start:end] for start, end in token[j]]
                page = answer[j]
                passage = ref_text[j]
                for sentence in list_of_sentences:
                    sentences.append(sentence)
                    passages.append(passage)
                    gold_pages.append(page)
                    passage_pages.append(page)
                    labels.append(1)

            list_text = text
            part_guesses = self.guesser.guess_wrong(list_text, n)
            for z in range(group_size):
                list_of_sentences = [text[z][start:end] for start, end in token[z]]
                part_guess = part_guesses[z]
                page = answer[z]
                cnt = 0
                for j in range(n):
                    neg_passage_page = part_guess[j][0]
                    if neg_passage_page == page:
                        continue
                    cnt += 1
                    neg_passage = self.wiki_lookup[neg_passage_page]['text']
                    for sentence in list_of_sentences:
                        sentences.append(sentence)
                        passages.append(neg_passage)
                        gold_pages.append(page)
                        passage_pages.append(neg_passage_page)
                        labels.append(0)
                    if cnt == n - 1:
                        break

        print('end processing dataset')
        print('total input size: {}'.format(len(labels)))
        new_data_dict = {'sentence': sentences, 'passage': passages, 'gold_page': gold_pages,
                         'passage_page': passage_pages, 'label': labels}
        answer_set = set()
        for answer in answers:
            answer_set.add(answer)
        print('total num of answers: {}'.format(len(answer_set)))
        if str == "train":
            filepath = 'models/train.pickle'
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'train_dict': new_data_dict
                }, f)
        else:
            filepath = 'models/eval.pickle'
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'eval_dict': new_data_dict
                }, f)
        return new_data_dict

    def preprocess_function(self, examples):
        inputs = self.tokenizer(
            examples['sentence'],
            examples['passage'],
            return_token_type_ids=True, padding='max_length', max_length=512, truncation=True, add_special_tokens=True,
        )
        inputs["labels"] = examples['label']
        return inputs

    def load(self, model_identifier: str = 'amberoad/bert-multilingual-passage-reranking-msmarco',
             model_path: str = None, max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
        if model_path is not None and exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_identifier, num_labels=2).to(device)
        print("self load complete")

    def train(self):
        if not exists('models/train.parquet'):
            train_questions = QantaDatabase('../data/qanta.train.2018.json').train_questions
            print('# of train questions: {}'.format(len(train_questions)))
            eval_questions = QantaDatabase('../data/qanta.dev.2018.json').dev_questions
            print('# of eval questions: {}'.format(len(eval_questions)))
            self.make_whole_dataset(train_questions, "train")
            self.make_whole_dataset(eval_questions, "eval")
            train_dataset = Dataset.from_parquet('models/train.parquet').shuffle(seed=42)
            eval_dataset = Dataset.from_parquet('models/eval.parquet').shuffle(seed=42)
            print("dataset complete")
            tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
            tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)
            tokenized_train_dataset.to_parquet('models/train.parquet')
            tokenized_eval_dataset.to_parquet('models/eval.parquet')
            print("preprocess complete")
        tokenized_train_dataset = Dataset.from_parquet('models/train.parquet')
        tokenized_eval_dataset = Dataset.from_parquet('models/eval.parquet')
        # tokenized_train_dataset = Dataset.from_dict(tokenized_train_questions)
        # tokenized_eval_dataset = Dataset.from_dict(tokenized_eval_questions)
        training_args = TrainingArguments(output_dir="models/new_guesser", save_total_limit=2, save_strategy="steps", num_train_epochs=4, report_to="wandb")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_train_dataset,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        self.model.save_pretrained('models/new_guesser')

    def get_sentence_rank(self, sentence, golden_page):
        n_ref_texts = 256
        guesses = self.guesser.guess([sentence], max_n_guesses=n_ref_texts)[0]
        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)
        with torch.no_grad():
            logits = torch.zeros(n_ref_texts)
            k = 64
            for i in tqdm(range((n_ref_texts + k - 1) // k)):
                inputs_B = ref_texts[i * k:(i + 1) * k]
                group_len = len(inputs_B)
                inputs_A = [sentence] * group_len
                model_inputs = self.tokenizer(
                    inputs_A, inputs_B, return_token_type_ids=True, padding='max_length', max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors='pt').to(device)

                model_outputs = self.model(**model_inputs)
                logits[i * k:i * k + group_len] = model_outputs.logits[:, 1]  # Label 1 means they are similar
            indices = torch.argsort(logits, dim=-1, descending=True)
            passage_set = set()
            for idx in indices:
                passage_page = guesses[idx][0]
                passage_set.add(passage_page)
                if passage_page == golden_page:
                    return len(passage_set)
            return n_ref_texts

    @staticmethod
    def update_metrics(metrics: np.array, line_no: int, rank: int):
        # metrics[line_no][0] += 1.0 * rank
        if rank == 1:
            metrics[line_no][1] += 1.0
        if rank <= 5:
            metrics[line_no][2] += 1.0
        if rank <= 20:
            metrics[line_no][3] += 1.0
        if rank <= 100:
            metrics[line_no][4] += 1.0

    def test(self):
        test_questions = QantaDatabase('../data/qanta.test.2018.json').guess_test_questions
        num_of_test_questions = len(test_questions)
        metrics = np.zeros((2, 5), dtype=float)

        for i in range(num_of_test_questions):
            test_question = test_questions[i]
            page = test_question.page
            first_sentence = test_question.first_sentence
            last_sentence = test_question.sentences[-1]
            first_rank = self.get_sentence_rank(first_sentence, page)
            last_rank = self.get_sentence_rank(last_sentence, page)
            self.update_metrics(metrics, 0, first_rank)
            self.update_metrics(metrics, 1, last_rank)
            print(i)
            print(metrics)

        metrics /= num_of_test_questions
        print('first avg rank: {}'.format(metrics[0][0]))
        print('first top 1: {}'.format(metrics[0][1]))
        print('first top 5: {}'.format(metrics[0][2]))
        print('first top 20: {}'.format(metrics[0][3]))
        print('first top 100: {}'.format(metrics[0][4]))
        print('last avg rank: {}'.format(metrics[1][0]))
        print('last top 1: {}'.format(metrics[1][1]))
        print('last top 5: {}'.format(metrics[1][2]))
        print('last top 20: {}'.format(metrics[1][3]))
        print('last top 100: {}'.format(metrics[1][4]))

    def test_squad(self, questions, pages):
        num_of_test_questions = len(questions)
        metrics = np.zeros((2, 5), dtype=float)

        for i in range(num_of_test_questions):
            question = questions[i]
            page = pages[i]
            rank = self.get_sentence_rank(question, page)
            self.update_metrics(metrics, 0, rank)
            print(i)
            print(metrics)

        metrics /= num_of_test_questions
        # print('first avg rank: {}'.format(metrics[0][0]))
        print('first top 1: {}'.format(metrics[0][1]))
        print('first top 5: {}'.format(metrics[0][2]))
        print('first top 20: {}'.format(metrics[0][3]))
        print('first top 100: {}'.format(metrics[0][4]))
        # print('last avg rank: {}'.format(metrics[1][0]))
        print('last top 1: {}'.format(metrics[1][1]))
        print('last top 5: {}'.format(metrics[1][2]))
        print('last top 20: {}'.format(metrics[1][3]))
        print('last top 100: {}'.format(metrics[1][4]))


class NewGuesserWithCT(BaseGuesser):

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None
        self.guesser = TfidfGuesser()
        self.guesser.load('new_models/tfidf.pickle')
        self.wiki_lookup = WikiLookup('../custom_data/wiki_lookup.json')

    def make_whole_dataset(self, examples, str):
        data_dict = {"text": [], "page": [], "first_sentence": [], "last_sentence": [], "tokenization": [],
                     "answer": []}
        for question in examples:
            data_dict["text"].append(question.text)
            data_dict["page"].append(question.page)
            data_dict["first_sentence"].append(question.first_sentence)
            data_dict["last_sentence"].append(question.sentences[-1])
            data_dict["tokenization"].append(question.tokenizations)
            data_dict["answer"].append(question.answer)

        questions = data_dict['text']  # list of text
        answers = data_dict['page']  # list of answer
        tokenization = data_dict['tokenization']  # list of tokenization
        ref_texts = []  # list of wiki passage
        for page in answers:
            ref_texts.append(self.wiki_lookup[page]['text'])

        l = len(questions)  # num of questions
        n = 2  # num of negative samples
        # max_length = 512
        # passage_len = 250     # max len of passage
        k = 1000  # process k at a time

        sentences = []  # 'sentence'
        passages = []  # 'passage'
        gold_pages = []  # 'gold_page'
        passage_pages = []  # 'passage_page'
        labels = []  # 'label'
        difficulties = []  # 'difficulty'

        print('start processing dataset')

        for i in tqdm(range((l + k - 1) // k)):
            text = questions[i * k: (i + 1) * k]  # group size
            token = tokenization[i * k: (i + 1) * k]  # group size
            answer = answers[i * k: (i + 1) * k]  # group size
            ref_text = ref_texts[i * k: (i + 1) * k]  # group size

            group_size = len(answer)
            for j in range(group_size):
                list_of_sentences = [text[j][start:end] for start, end in token[j]]
                page = answer[j]
                passage = ref_text[j]
                for idx, sentence in enumerate(list_of_sentences):
                    sentences.append(sentence)
                    passages.append(passage)
                    gold_pages.append(page)
                    passage_pages.append(page)
                    labels.append(1)
                    difficulties.append(idx + 1)

            list_text = text
            part_guesses = self.guesser.guess_wrong(list_text, n)
            for z in range(group_size):
                list_of_sentences = [text[z][start:end] for start, end in token[z]]
                part_guess = part_guesses[z]
                page = answer[z]
                cnt = 0
                for j in range(n):
                    neg_passage_page = part_guess[j][0]
                    if neg_passage_page == page:
                        continue
                    cnt += 1
                    neg_passage = self.wiki_lookup[neg_passage_page]['text']
                    for idx, sentence in enumerate(list_of_sentences):
                        sentences.append(sentence)
                        passages.append(neg_passage)
                        gold_pages.append(page)
                        passage_pages.append(neg_passage_page)
                        labels.append(0)
                        difficulties.append(idx + 1)
                    if cnt == n - 1:
                        break

        print('end processing dataset')
        print('total input size: {}'.format(len(labels)))
        new_data_dict = {'sentence': sentences, 'passage': passages, 'gold_page': gold_pages,
                         'passage_page': passage_pages, 'label': labels, 'difficulty': difficulties}
        answer_set = set()
        for answer in answers:
            answer_set.add(answer)
        print('total num of answers: {}'.format(len(answer_set)))
        pd_frame = pd.DataFrame.from_dict(new_data_dict)
        if str == "train":
            new_data_frame = pd.DataFrame.from_dict(new_data_dict)
            # new_data_frame.sort_values(by=['difficulty'], ascending=False)
            simple_frame = new_data_frame.loc[new_data_frame['difficulty'] >= 6]
            medium_frame = new_data_frame.loc[(new_data_frame['difficulty'] >= 3) & (new_data_frame['difficulty'] <= 5)]
            hard_frame = new_data_frame.loc[(new_data_frame['difficulty'] >= 1) & (new_data_frame['difficulty'] <= 2)]
            simple_frame.to_parquet('new_models/simple.parquet')
            medium_frame.to_parquet('new_models/medium.parquet')
            hard_frame.to_parquet('new_models/hard.parquet')
        else:
            pd_frame.to_parquet('new_models/eval.parquet')

    def make_whole_data_dict(self, data_dict, str):
        questions = data_dict['text']  # list of text
        answers = data_dict['page']  # list of answer
        tokenization = data_dict['tokenization']  # list of tokenization
        ref_texts = []  # list of wiki passage
        for page in answers:
            ref_texts.append(self.wiki_lookup[page]['text'])

        l = len(questions)  # num of questions
        n = 2  # num of negative samples
        # max_length = 512
        # passage_len = 250       # max len of passage
        k = 1000  # process k at a time

        sentences = []  # 'sentence'
        passages = []  # 'passage'
        gold_pages = []  # 'gold_page'
        passage_pages = []  # 'passage_page'
        labels = []  # 'label'
        difficulties = []  # 'difficulty'

        print('start processing dataset')

        for i in tqdm(range((l + k - 1) // k)):
            text = questions[i * k: (i + 1) * k]  # group size
            token = tokenization[i * k: (i + 1) * k]  # group size
            answer = answers[i * k: (i + 1) * k]  # group size
            ref_text = ref_texts[i * k: (i + 1) * k]  # group size

            group_size = len(answer)
            for j in range(group_size):
                list_of_sentences = [text[j][start:end] for start, end in token[j]]
                passage = ref_text[j]
                page = answer[j]
                for idx, sentence in enumerate(list_of_sentences):
                    sentences.append(sentence)
                    passages.append(passage)
                    gold_pages.append(page)
                    passage_pages.append(page)
                    labels.append(1)
                    difficulties.append(idx + 1)

            list_text = text
            part_guesses = self.guesser.guess_wrong(list_text, n)
            for z in range(group_size):
                list_of_sentences = [text[z][start:end] for start, end in token[z]]
                part_guess = part_guesses[z]
                page = answer[z]
                cnt = 0
                for j in range(n):
                    neg_passage_page = part_guess[j][0]
                    if neg_passage_page == page:
                        continue
                    cnt += 1
                    neg_passage = self.wiki_lookup[neg_passage_page]['text']
                    for idx, sentence in enumerate(list_of_sentences):
                        sentences.append(sentence)
                        passages.append(neg_passage)
                        gold_pages.append(page)
                        passage_pages.append(neg_passage_page)
                        labels.append(0)
                        difficulties.append(idx + 1)
                    if cnt == n - 1:
                        break
        filepath = 'new_models/passage.pickle'
        print('end processing dataset')
        print('total input size: {}'.format(len(labels)))
        new_data_dict = {'sentence': sentences, 'passage': passages, 'gold_page': gold_pages,
                         'passage_page': passage_pages, 'label': labels, 'difficulty': difficulties}
        answer_set = set()
        for answer in answers:
            answer_set.add(answer)
        print('total num of answers: {}'.format(len(answer_set)))

        if str == "train":
            new_data_frame = pd.DataFrame.from_dict(new_data_dict)
            new_data_frame.sort_values(by=['difficulty'], ascending=False)
            simple_frame = new_data_frame.loc[new_data_frame['difficulty'] >= 6]
            medium_frame = new_data_frame.loc[(new_data_frame['difficulty'] >= 3) & (new_data_frame['difficulty'] <= 5)]
            hard_frame = new_data_frame.loc[(new_data_frame['difficulty'] >= 1) & (new_data_frame['difficulty'] <= 2)]
            simple_dict = simple_frame.to_dict('list')
            medium_dict = medium_frame.to_dict('list')
            hard_dict = hard_frame.to_dict('list')

            filepath = 'new_models/train.pickle'
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'simple_dict': simple_dict,
                    'medium_dict': medium_dict,
                    'hard_dict': hard_dict
                }, f)
            return simple_dict, medium_dict, hard_dict
        else:
            filepath = 'new_models/eval.pickle'
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'eval_dict': new_data_dict
                }, f)
            return new_data_dict

    def preprocess_function(self, examples):
        inputs = self.tokenizer(
            examples['sentence'],
            examples['passage'],
            return_token_type_ids=True, padding='max_length', max_length=512, add_special_tokens=True, truncation=True
        )
        inputs["labels"] = examples['label']
        return inputs

    def load(self, model_identifier: str = 'amberoad/bert-multilingual-passage-reranking-msmarco',
             model_path: str = None, max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
        if model_path is not None and exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_identifier, num_labels=2).to(device)
        print("self load complete")

    def train(self):
        if not exists('new_models/eval.parquet'):
            train_questions = QantaDatabase('../data/qanta.train.2018.json').train_questions
            print('# of train questions: {}'.format(len(train_questions)))
            eval_questions = QantaDatabase('../data/qanta.dev.2018.json').dev_questions
            print('# of eval questions: {}'.format(len(eval_questions)))
            self.make_whole_dataset(train_questions, "train")
            self.make_whole_dataset(eval_questions, "eval")
            simple_train_dataset = Dataset.from_parquet('new_models/simple.parquet').shuffle(seed=42)
            medium_train_dataset = Dataset.from_parquet('new_models/medium.parquet').shuffle(seed=42)
            hard_train_dataset = Dataset.from_parquet('new_models/hard.parquet').shuffle(seed=42)
            eval_dataset = Dataset.from_parquet('new_models/eval.parquet').shuffle(seed=42)
            print("dataset complete")
            tokenized_simple_train_dataset = simple_train_dataset.map(self.preprocess_function, batched=True)
            tokenized_medium_train_dataset = medium_train_dataset.map(self.preprocess_function, batched=True)
            tokenized_hard_train_dataset = hard_train_dataset.map(self.preprocess_function, batched=True)
            tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)
            print("preprocess complete")
            tokenized_simple_train_dataset.to_parquet('new_models/simple.parquet')
            tokenized_medium_train_dataset.to_parquet('new_models/medium.parquet')
            tokenized_hard_train_dataset.to_parquet('new_models/hard.parquet')
            tokenized_eval_dataset.to_parquet('new_models/eval.parquet')
        tokenized_simple_train_dataset = Dataset.from_parquet('new_models/simple.parquet')
        tokenized_medium_train_dataset = Dataset.from_parquet('new_models/medium.parquet')
        tokenized_hard_train_dataset = Dataset.from_parquet('new_models/hard.parquet')
        tokenized_eval_dataset = Dataset.from_parquet('new_models/eval.parquet')

        st = tokenized_simple_train_dataset
        mt = concatenate_datasets([st, tokenized_medium_train_dataset])
        ht = concatenate_datasets([mt, tokenized_hard_train_dataset])

        # tokenized_train_dataset = Dataset.from_dict(tokenized_train_questions)
        # tokenized_eval_dataset = Dataset.from_dict(tokenized_eval_questions)
        training_args = TrainingArguments(output_dir="new_models/new_guesser", save_total_limit=2, num_train_epochs=2, save_strategy="steps", report_to="wandb")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=st,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        trainer = Trainer(model=self.model, args=training_args, train_dataset=mt,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        trainer = Trainer(model=self.model, args=training_args, train_dataset=ht,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        self.model.save_pretrained('new_models/new_guesser')

    def get_sentence_rank(self, sentence, golden_page):
        n_ref_texts = 256
        guesses = self.guesser.guess([sentence], max_n_guesses=n_ref_texts)[0]
        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)
        with torch.no_grad():
            logits = torch.zeros(n_ref_texts)
            k = 64
            for i in tqdm(range((n_ref_texts + k - 1) // k)):
                inputs_B = ref_texts[i * k:(i + 1) * k]
                group_len = len(inputs_B)
                inputs_A = [sentence] * group_len
                model_inputs = self.tokenizer(
                    inputs_A, inputs_B, return_token_type_ids=True, padding='max_length', max_length=512,
                    truncation=True,
                    add_special_tokens=True,
                    return_tensors='pt').to(device)

                model_outputs = self.model(**model_inputs)
                logits[i * k:i * k + group_len] = model_outputs.logits[:, 1]  # Label 1 means they are similar
            indices = torch.argsort(logits, dim=-1, descending=True)
            passage_set = set()
            for idx in indices:
                passage_page = guesses[idx][0]
                passage_set.add(passage_page)
                if passage_page == golden_page:
                    return len(passage_set)
            return n_ref_texts

    @staticmethod
    def update_metrics(metrics: np.array, line_no: int, rank: int):
        # metrics[line_no][0] += 1.0 * rank
        if rank == 1:
            metrics[line_no][1] += 1.0
        if rank <= 5:
            metrics[line_no][2] += 1.0
        if rank <= 20:
            metrics[line_no][3] += 1.0
        if rank <= 100:
            metrics[line_no][4] += 1.0

    def test(self):
        test_questions = QantaDatabase('../data/qanta.test.2018.json').guess_test_questions
        num_of_test_questions = len(test_questions)
        metrics = np.zeros((2, 5), dtype=float)

        for i in range(num_of_test_questions):
            test_question = test_questions[i]
            page = test_question.page
            first_sentence = test_question.first_sentence
            last_sentence = test_question.sentences[-1]
            first_rank = self.get_sentence_rank(first_sentence, page)
            last_rank = self.get_sentence_rank(last_sentence, page)
            self.update_metrics(metrics, 0, first_rank)
            self.update_metrics(metrics, 1, last_rank)
            print(i)
            print(metrics)

        metrics /= num_of_test_questions
        # print('first avg rank: {}'.format(metrics[0][0]))
        print('first top 1: {}'.format(metrics[0][1]))
        print('first top 5: {}'.format(metrics[0][2]))
        print('first top 20: {}'.format(metrics[0][3]))
        print('first top 100: {}'.format(metrics[0][4]))
        # print('last avg rank: {}'.format(metrics[1][0]))
        print('last top 1: {}'.format(metrics[1][1]))
        print('last top 5: {}'.format(metrics[1][2]))
        print('last top 20: {}'.format(metrics[1][3]))
        print('last top 100: {}'.format(metrics[1][4]))

    def test_squad(self, questions, pages):
        num_of_test_questions = len(questions)
        metrics = np.zeros((2, 5), dtype=float)

        for i in range(num_of_test_questions):
            question = questions[i]
            page = pages[i]
            rank = self.get_sentence_rank(question, page)
            self.update_metrics(metrics, 0, rank)
            print(i)
            print(metrics)

        metrics /= num_of_test_questions
        # print('first avg rank: {}'.format(metrics[0][0]))
        print('first top 1: {}'.format(metrics[0][1]))
        print('first top 5: {}'.format(metrics[0][2]))
        print('first top 20: {}'.format(metrics[0][3]))
        print('first top 100: {}'.format(metrics[0][4]))
        # print('last avg rank: {}'.format(metrics[1][0]))
        print('last top 1: {}'.format(metrics[1][1]))
        print('last top 5: {}'.format(metrics[1][2]))
        print('last top 20: {}'.format(metrics[1][3]))
        print('last top 100: {}'.format(metrics[1][4]))


class ReRanker(BaseReRanker):
    """A Bert based Reranker that consumes a reference passage and a question as input text and predicts the similarity score: 
        likelihood for the passage to contain the answer to the question.

    Task: Load any pretrained BERT-based and finetune on QuizBowl (or external) examples to enable this model to predict scores 
        for each reference text for an input question, and use that score to rerank the reference texts.

    Hint: Try to create good negative samples for this binary classification / score regression task.

    Documentation Links:

        Pretrained Tokenizers:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/model#transformers.PreTrainedModel.from_pretrained

        BERT for Sequence Classification:
            https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/bert#transformers.BertForSequenceClassification

        SequenceClassifierOutput:
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput

        Fine Tuning BERT for Seq Classification:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#sequence-classification-with-imdb-reviews

        Passage Reranking:
            https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco
    """

    def __init__(self) -> None:
        self.tokenizer = None
        self.model = None

    def make_whole_data_dict(self, data_dict):
        questions = data_dict['text']
        answers = data_dict['page']
        first_sentences = data_dict['first_sentence']
        ref_texts = []
        for page in answers:
            ref_texts.append(self.wiki_lookup[page]['text'])

        k = 1000
        l = len(questions)
        n = int((l + k - 1) / k)

        for i in range(n):
            if i == n - 1:
                part_questions = questions[i * k: l]
                part_answers = answers[i * k: l]
                part_first_sentences = first_sentences[i * k: l]
            else:
                part_questions = questions[i * k:(i + 1) * k]
                part_answers = answers[i * k:(i + 1) * k]
                part_first_sentences = first_sentences[i * k:(i + 1) * k]

            part_guesses = self.guesser.guess_wrong(part_questions, 2)
            part_l = len(part_guesses)
            for j in range(part_l):
                first_sentences.append(part_first_sentences[j])
                if part_guesses[j][0][0] == part_answers[j]:
                    ref_texts.append(self.wiki_lookup[part_guesses[j][1][0]]['text'])
                else:
                    ref_texts.append(self.wiki_lookup[part_guesses[j][0][0]]['text'])
            print('{} in {} completed'.format(i, n))

        labels = [1] * l + [0] * l
        new_data_dict = {'first_sentence': first_sentences, 'ref_text': ref_texts, 'labels': labels}
        return new_data_dict

    def preprocess_function(self, examples):
        inputs = self.tokenizer(
            examples['first_sentence'],
            examples['ref_text'],
            return_token_type_ids=True, padding='max_length', max_length=512, truncation=True, add_special_tokens=True,
        )
        inputs["labels"] = examples['labels']
        return inputs

    def load(self, model_identifier: str, model_path: str = None, max_model_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, model_max_length=max_model_length)
        if model_path is not None and exists(model_path):
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_identifier, num_labels=2).to(device)
        print("self load complete")

    def train(self):
        """Fill this method with code that finetunes Sequence Classification task on QuizBowl questions and passages.
        Feel free to change and modify the signature of the method to suit your needs."""

        # dataset = load_dataset('json', data_files={'train':'../data/qanta.train.2018.json', 'validation':'../data/qanta.dev.2018.json'})
        # train_dataset = dataset['train']
        # eval_dataset = dataset['eval']
        # tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        # tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)
        # eval_dataset = load_dataset('json', data_files='../data/qanta.dev.2018.json')
        self.wiki_lookup = WikiLookup('/custom_data/wiki_lookup.2018.json')
        self.guesser = TfidfGuesser()
        print('Loading the Guesser model...')
        self.guesser.load('models/tfidf.pickle')
        print('Load complete!')

        train_questions = QantaDatabase('../data/qanta.train.2018.json').train_questions
        eval_questions = QantaDatabase('../data/qanta.dev.2018.json').dev_questions
        train_dict = self.make_whole_data_dict(make_data_dict(train_questions))
        eval_dict = self.make_whole_data_dict(make_data_dict(eval_questions))
        train_dataset = Dataset.from_dict(train_dict).shuffle(seed=42)
        eval_dataset = Dataset.from_dict(eval_dict).shuffle(seed=42)
        print("dataset complete")
        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)
        print("preprocess complete")
        # tokenized_train_dataset = Dataset.from_dict(tokenized_train_questions)
        # tokenized_eval_dataset = Dataset.from_dict(tokenized_eval_questions)
        training_args = TrainingArguments(output_dir="/mnt/disks/models/new_reranker", save_total_limit=1,
                                          load_best_model_at_end=True, save_strategy="no")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_train_dataset,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        self.model.save_pretrained('/mnt/disks/models/new_reranker')

    def get_best_document(self, question: str, ref_texts: List[str]) -> int:
        """Selects the best reference text from a list of reference text for each question."""

        with torch.no_grad():
            n_ref_texts = len(ref_texts)
            inputs_A = [question] * n_ref_texts
            inputs_B = ref_texts

            model_inputs = self.tokenizer(
                inputs_A, inputs_B, return_token_type_ids=True, padding='max_length', max_length=512, truncation=True,
                add_special_tokens=True,
                return_tensors='pt').to(device)

            model_outputs = self.model(**model_inputs)
            logits = model_outputs.logits[:, 1]  # Label 1 means they are similar

            return torch.argmax(logits, dim=-1)


class Retriever:
    """The component that indexes the documents and retrieves the top document from an index for an input open-domain question.
    
    It uses two systems:
     - Guesser that fetches top K documents for an input question, and
     - ReRanker that then reranks these top K documents by comparing each of them with the question to produce a similarity score."""

    def __init__(self, guesser: BaseGuesser, reranker: BaseReRanker, wiki_lookup: Union[str, WikiLookup],
                 max_n_guesses=10) -> None:
        if isinstance(wiki_lookup, str):
            self.wiki_lookup = WikiLookup(wiki_lookup)
        else:
            self.wiki_lookup = wiki_lookup
        self.guesser = guesser
        self.reranker = reranker
        self.max_n_guesses = max_n_guesses

    def retrieve_answer_document(self, question: str, disable_reranking=False) -> str:
        """Returns the best guessed page that contains the answer to the question."""
        guesses = self.guesser.guess([question], max_n_guesses=self.max_n_guesses)[0]

        if disable_reranking:
            _, best_page = max((score, page) for page, score in guesses)
            return best_page

        ref_texts = []
        for page, score in guesses:
            doc = self.wiki_lookup[page]['text']
            ref_texts.append(doc)

        best_doc_id = self.reranker.get_best_document(question, ref_texts)
        return guesses[best_doc_id][0]


class AnswerExtractor:
    """Load a huggingface model of type transformers.AutoModelForQuestionAnswering and finetune it for QuizBowl questions.

    Documentation Links:

        Extractive QA: 
            https://huggingface.co/docs/transformers/v4.16.2/en/task_summary#extractive-question-answering

        QA Pipeline: 
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/pipelines#transformers.QuestionAnsweringPipeline

        QAModelOutput: 
            https://huggingface.co/docs/transformers/v4.16.2/en/main_classes/output#transformers.modeling_outputs.QuestionAnsweringModelOutput

        Finetuning Answer Extraction:
            https://huggingface.co/docs/transformers/master/en/custom_datasets#question-answering-with-squad
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None

    def make_whole_data_dict(self, data_dict):
        pages = data_dict['page']
        answers = data_dict['answer']
        first_sentences = data_dict['first_sentence']
        last_sentences = data_dict['last_sentence']

        questions = []
        ref_texts = []
        split_answers = []

        l = len(pages)
        for i in range(l):
            page = pages[i]
            answer_set = generate_gold_answers(answers[i])
            first_sentence = first_sentences[i]
            last_sentence = last_sentences[i]
            ref_text = self.wiki_lookup[page]['text']
            for answer in answer_set:
                if ref_text.lower().find(answer) != -1:
                    answer_dict = {'answer_start': [ref_text.lower().find(answer)], 'text': [answer]}
                    questions.append(first_sentence)
                    ref_texts.append(ref_text)
                    split_answers.append(answer_dict)
                    questions.append(last_sentence)
                    ref_texts.append(ref_text)
                    split_answers.append(answer_dict)
        new_data_dict = {'questions': questions, 'ref_texts': ref_texts, 'answers': split_answers}
        return new_data_dict

    def preprocess_function(self, examples):
        inputs = self.tokenizer(
            examples['questions'],
            examples['ref_texts'],
            return_token_type_ids=True, padding='max_length', max_length=512, truncation=True, add_special_tokens=True,
            return_offsets_mapping=True,
        )

        start_positions = []
        end_positions = []
        offset_mapping = inputs.pop("offset_mapping")
        answers = examples['answers']

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)
            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def load(self, model_identifier: str, model_path: str = None, max_model_length: int = 512):

        # You don't need to re-train the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, max_model_length=max_model_length)

        # Finetune this model for QuizBowl questions
        if model_path is not None and exists(model_path):
            self.model = AutoModelForQuestionAnswering.from_pretrained(model_path).to(device)
        else:
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                model_identifier).to(device)

    def train(self):
        """Fill this method with code that finetunes Answer Extraction task on QuizBowl examples.
        Feel free to change and modify the signature of the method to suit your needs."""
        self.wiki_lookup = WikiLookup('/custom_data/wiki_lookup.2018.json')
        train_questions = QantaDatabase('../data/qanta.train.2018.json').train_questions
        eval_questions = QantaDatabase('../data/qanta.dev.2018.json').dev_questions
        train_dict = self.make_whole_data_dict(make_data_dict(train_questions))
        eval_dict = self.make_whole_data_dict(make_data_dict(eval_questions))
        train_dataset = Dataset.from_dict(train_dict).shuffle(seed=42)
        eval_dataset = Dataset.from_dict(eval_dict).shuffle(seed=42)
        print("dataset complete")
        tokenized_train_dataset = train_dataset.map(self.preprocess_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.preprocess_function, batched=True)
        print("preprocess complete")
        # tokenized_train_dataset = Dataset.from_dict(tokenized_train_questions)
        # tokenized_eval_dataset = Dataset.from_dict(tokenized_eval_questions)
        training_args = TrainingArguments(output_dir="/mnt/disks/models/new_extractor", save_total_limit=1,
                                          load_best_model_at_end=True, save_strategy="no")
        trainer = Trainer(model=self.model, args=training_args, train_dataset=tokenized_train_dataset,
                          eval_dataset=tokenized_eval_dataset)
        trainer.train()
        self.model.save_pretrained('/mnt/disks/models/new_extractor')

    def extract_answer(self, question: Union[str, List[str]], ref_text: Union[str, List[str]]) -> List[str]:
        """Takes a (batch of) questions and reference texts and returns an answer text from the 
        reference which is answer to the input question.
        """
        with torch.no_grad():
            model_inputs = self.tokenizer(
                question, ref_text, return_tensors='pt', truncation=True, padding='max_length', max_length=512,
                return_token_type_ids=True, add_special_tokens=True).to(device)
            outputs = self.model(**model_inputs)
            input_tokens = model_inputs['input_ids']
            start_index = torch.argmax(outputs.start_logits, dim=-1)
            end_index = torch.argmax(outputs.end_logits, dim=-1)

            answer_ids = [tokens[s:e + 1] for tokens, s, e in zip(input_tokens, start_index, end_index)]

            return self.tokenizer.batch_decode(answer_ids)
