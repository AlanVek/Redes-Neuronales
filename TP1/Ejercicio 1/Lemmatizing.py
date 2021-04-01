from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
import nltk
from nltk.corpus import wordnet

twenty_train = fetch_20newsgroups(subset='train', shuffle=True, remove = ('headers', 'footers'))

twenty_test = fetch_20newsgroups(subset = 'test', remove = ('headers', 'footers'))

wnl = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

analyzer = CountVectorizer(stop_words = 'english').build_analyzer()
def tokenizer(words): return (wnl.lemmatize(word, pos = get_wordnet_pos(word)) for word in analyzer(words))


train = []
test = []

i = 1
tot = twenty_train['data'].size
for news in twenty_train['data']:
    print(f'{i} of {tot}')
    i += 1
    train.append(' '.join(tokenizer(news)))

i = 1
tot = twenty_test['data'].size
for news in twenty_test['data']:
    print(f'{i} of {tot}')
    i += 1
    test.append(' '.join(tokenizer(news)))

with open('test_data.txt', 'wt') as file:
    for news in test: file.write(news + '\n')

with open('train_data.txt', 'wt') as file:
    for news in train: file.write(news + '\n')