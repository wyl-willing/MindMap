import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import json
import re


re3 = r"<CLS>(.*?)<SEP>"
docs_dir = './data/chatdoctor5k/document'

docs = []
for file in os.listdir(docs_dir):
    with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
        doc = f.read()
        docs.append(doc)
questions = []
with open("./data/chatdoctor5k/NER_chatgpt.json", "r") as f:
    for line in f.readlines():
        x = json.loads(line)
        input = x["qustion_output"]
        input = input.replace("\n","")
        input = input.replace("<OOS>","<EOS>")
        input = input.replace(":","") + "<END>"
        input_text = re.findall(re3,input)
        if input_text == []:
            continue
        questions.append(input_text[0])
        
sentences = [doc.split() for doc in docs + questions]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
model.save("./data/chatdoctor5k/word2vec.model")
