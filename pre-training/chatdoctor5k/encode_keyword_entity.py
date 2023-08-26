import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# load data
import json, pdb, re

with open(".data/chatdoctor5k/entity2id.txt","r") as f:
    entities = f.readlines()
    entities = [entity.strip().split()[0].replace("_"," ") for entity in entities]

keywords = set([])

with open("dataset3_ner.json", "r") as f:
    for line in f.readlines():
        x = json.loads(line)
      
        question_kg = x["question_kg"]
        question_kg = question_kg.replace("\n","")
        kws = question_kg.split(", ")
        
        [keywords.add(kw) for kw in kws]
keywords = list(keywords)

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
model.to("cuda")

# encode entities
embeddings = model.encode(entities, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
entity_emb_dict = {
    "entities": entities,
    "embeddings": embeddings,
}
import pickle
with open("entity_embeddings.pkl", "wb") as f:
    pickle.dump(entity_emb_dict, f)

# encode keywords
embeddings = model.encode(keywords, batch_size=1024, show_progress_bar=True, normalize_embeddings=True)
keyword_emb_dict = {
    "keywords": keywords,
    "embeddings": embeddings,
}
import pickle
with open("keyword_embeddings.pkl", "wb") as f:
    pickle.dump(keyword_emb_dict, f)

print("done!")
