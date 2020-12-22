# -*- coding: utf-8 -*-
import os
import nltk
nltk.download('punkt')
SENT_DETECTOR = nltk.data.load('tokenizers/punkt/english.pickle')

from emoji import UNICODE_EMOJI
from os.path import dirname, abspath
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


sample_file = "christmas.txt"

if __name__ == '__main__':
    d = dirname(abspath(__file__)) # /../EmojiProject
    data_files = os.path.join(d, "data/original")
    filepath = os.path.join(data_files, sample_file)
    
    
    #open sample text
    openfile = open(filepath, encoding="utf8")
    text = openfile.read()
    #initial tokenization
    words = word_tokenize(text)
    
    #Very basic parsing of emotes
    #Create list to store tokenized words/emoji
    parsed_text = []
    emoji_counter = 0
    
    files = os.listdir(data_files)
    for file in files:
        filepath = os.path.join(data_files, file)
        openfile = open(filepath, encoding="utf8")
        text = openfile.read()
        words = word_tokenize(text)
        for word in words:
            word = str.lower(word)
            emojis = []
            #search for emoji in given word
            for char in word:
                if char in UNICODE_EMOJI:
                    if char is not '🙂':
                        emojis.append(char)
                        emoji_counter+=1
            #add each emoji to parsed storage and remove from word
            for char in emojis:
                word = word.replace(char, '')
                parsed_text.append(char)
            #reduce blank entries in parsed text
            if(len(word) > 1):
                parsed_text.append(word)
            
    #word2vec copy-pasta
    # train model
    model = Word2Vec([parsed_text], min_count=1)
    # summarize the loaded model
    print(model)
    # summarize vocabulary
    words = list(model.wv.vocab)
    print(words)
    # access vector for one word
    print(model['🍆'])
    # save model
    model.save('model.bin')
    # load model
    new_model = Word2Vec.load('model.bin')
    print(new_model)
    
    print(emoji_counter)


    #word2vec display
    #display first 100 vocab, not necessarily 100 most important
    font_name = os.path.join(d, "OpenSansEmoji.ttf")
    prop = FontProperties(fname=font_name)
    plt.rcParams['font.family'] = prop.get_family()
    # define training dataS
    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    plt.scatter(result[:100, 0], result[:100, 1])
    for i, word in enumerate(words):
        if(i > 100):
            break
        plt.annotate(word, xy=(result[i, 0], result[i, 1]),fontproperties=prop)
    plt.show()
    
    