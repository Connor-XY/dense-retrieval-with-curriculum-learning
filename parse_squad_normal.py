import json
from qbdata import WikiLookup, QantaDatabase, Question
wiki_lookup = WikiLookup('../custom_data/wiki_lookup.json')
f = open('../data/train-v2.0.json')
data = json.load(f)
# print(data.keys())
data = data["data"]
n = len(data)
page_set = set()
pages = []
questions = []
for i in range(n):
    subject = data[i]
    # print(subject.keys())
    title = subject["title"]
    if len(title) == len(wiki_lookup[title]['text']):
        continue
    paragraphs = subject["paragraphs"]
    for paragraph in paragraphs:
        # print(paragraph.keys())
        qas = paragraph['qas']
        context = paragraph['context']
        for qs in qas:
            if 'plausible_answers' in qs.keys():
                continue
            page_set.add(title)
            pages.append(title)
            questions.append(qs['question'])

print(len(page_set))
print(page_set)
print(len(pages))
print(len(questions))

candidate_pages = []
candidate_ref_texts = []
for page in page_set:
    candidate_pages.append(page)
    candidate_ref_texts.append(wiki_lookup[page]['text'])

from models import NewGuesser, NewGuesserWithCT
guesser = NewGuesser()
guesser.passage_texts = candidate_ref_texts
guesser.answers = candidate_pages
guesser.load(model_path='models/new_guesser')
guesser.test_squad(questions, pages)
