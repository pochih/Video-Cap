from __future__ import print_function

import sys
from os import listdir
from os.path import join, isfile

import json
from collections import Counter
from gensim.models import word2vec, KeyedVectors

MIN_WORD_COUNT = 3
WORD_VECTOR_SIZE = 300

video_path = join(sys.argv[1], 'training_data/video')
feat_path = join(sys.argv[1], 'training_data/feat')

videos = [f for f in listdir(video_path) if isfile(join(video_path, f))]
feats = [f for f in listdir(feat_path) if isfile(join(feat_path, f))]

with open(join(sys.argv[1], 'training_label.json')) as json_data:
    training_labels = json.load(json_data)

with open(join(sys.argv[1], 'testing_public_label.json')) as json_data:
    testing_labels = json.load(json_data)

with open('model/corpus.txt', 'w') as out:
	for label in training_labels:
		for sentence in label['caption']:
			sentence = '<BOS> ' + sentence.lower()[:-1] + ' <EOS>'
			out.write(sentence + ' ')
	for label in testing_labels:
		for sentence in label['caption']:
			sentence = '<BOS> ' + sentence.lower()[:-1] + ' <EOS>'
			out.write(sentence + ' ')

from operator import itemgetter
import heapq
def least_common_values(counter, to_find=None):
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))

# build word vector, type: word embedding
print("Building embedded word vector...")
corpus = word2vec.Text8Corpus("model/corpus.txt")
word_vector = word2vec.Word2Vec(corpus, size=WORD_VECTOR_SIZE)
word_vector.wv.save_word2vec_format("model/word_vector.bin", binary=True)
#word_vector = KeyedVectors.load_word2vec_format('model/word_vector.bin', binary=True)

# build word dictionary  
print("Building word dictionary...")
word_dict = Counter(open("model/corpus.txt", 'r').read().split())
# testing_data_word_dict = Counter(open("model/testing_data_words.txt", 'r').read().split())
# print("Top 20 most appearance", word_dict.most_common(20))
# print("Top 20 least appearance", least_common_values(word_dict, 20))
print("words num", len(word_dict.keys()))
for word in list(word_dict):
    if word_dict[word] < MIN_WORD_COUNT:
        del word_dict[word]
print("words num after filtering", len(word_dict.keys()))
print("Top 20 least appearance", least_common_values(word_dict, 20))