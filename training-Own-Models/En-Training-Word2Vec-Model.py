#-*- codeing=utf-8 -*-
#@time: 2020/9/6 23:12
#@Author: Shang-gang Lee

from gensim.test.utils import datapath
from gensim import utils

class MyCorpus(object):

    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        corpus_path = datapath('lee_background.cor')
        for line in open(corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)

import gensim.models
sentences = MyCorpus()
sentences=[i for i in sentences]
print(type(sentences))
w2v_model = gensim.models.Word2Vec(min_count=1,
                     window=2,
                     size=300,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20)
w2v_model.build_vocab(sentences)
w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=10)
print(w2v_model.wv.most_similar(positive=["hundred"]))
print(w2v_model.wv["hundred"].shape)
