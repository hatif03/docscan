import spacy
from spacy.tokens import DocBin
import pickle

nlp = spacy.blank("en")

# Load data
training_data = pickle.load(open('./processed_data/train.pickle', 'rb'))
test_data = pickle.load(open('./processed_data/test.pickle', 'rb'))

# the DocBin will store the example documents
db = DocBin()
for text, annotations in training_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db.add(doc)
db.to_disk("./processed_data/train.spacy")


# the DocBin will store the example documents
db_test = DocBin()
for text, annotations in test_data:
    doc = nlp(text)
    ents = []
    for start, end, label in annotations['entities']:
        span = doc.char_span(start, end, label=label)
        ents.append(span)
    doc.ents = ents
    db_test.add(doc)
db_test.to_disk("./processed_data/test.spacy")