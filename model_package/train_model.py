"""
Given a list of parsed file directories, train the word2vec model

@author: Owner
"""
import os, sys
from os.path import dirname, abspath
sys.path.append("/models")

import json
import split_method
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings("ignore")

from emoji import UNICODE_EMOJI
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def train_model(files):
    sentences = []
    #recover file contents, should be tokenized list of words/emojis
    for file in files:
        with open(file) as json_file:
            f = json.load(json_file)
            text = f["tokenized"]
        sentences.append(text)
        
    #word2vec copy-pasta
    # train model
    model = Word2Vec(sentences, min_count=5, sg=0)# sg0 = CBOW, sg1 = Skipgram
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print(model['😂'])
    # save model
    model.save('model.bin')

    #word2vec display
    #display first 100 vocab, not necessarily 100 most important
    d = dirname(dirname(abspath(__file__))) # /../EmojiProject
    font_name = os.path.join(d, "OpenSansEmoji.ttf")
    prop = FontProperties(fname=font_name)
    plt.rcParams['font.family'] = prop.get_family()
    plt.figure(dpi=1000)
    # define training dataS
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(words):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]),fontproperties=prop)
    plt.show()
    
if __name__ == '__main__':
    data = split_method.Data()
    training, testing = data.split_data(0.8)
    train_model(training)