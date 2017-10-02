from sklearn.feature_extraction.text import TfidfVectorizer
import re
import os
import pickle

cur_dir = os.path.dirname(os.path.abspath('__file__'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|p)')
    text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) \
            + ' '.join(empticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = TfidfVectorizer()
