#-*- coding: utf-8 -*-

from model import Video_Caption_Generator
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
import time
import json
# import matplotlib.pyplot as plt
import bleu_eval

#=====================================================
# Global Parameters
#=====================================================
video_train_feat_path = './data/train_features'
video_test_feat_path = './data/test_features'

video_train_data_path = './data/video_corpus.csv'
video_test_data_path = './data/video_corpus.csv'

testing_data = './data/testing_id.txt'
testing_label = './data/testing_public_label.json'
# model_path = './models'
default_model_path = './models/model-990'

#=====================================================
# Train Parameters
#=====================================================
dim_image = 4096
dim_hidden = 1000

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80

batch_size = 50

def test(model_path=default_model_path):
    test_videos = open(testing_data, 'r').read().split('\n')[:-1]
    with open(testing_label) as data_file:
        test_labels = json.load(data_file)

    ixtoword = pd.Series(np.load('./data/ixtoword.npy').tolist())

    bias_init_vector = np.load('./data/bias_init_vector.npy')

    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step,
            bias_init_vector=bias_init_vector)

    video_tf, video_mask_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    try:
        print '\n=== Use model', model_path, '===\n'
        saver.restore(sess, model_path)
    except:
        print '\nUse default model\n'
        saver.restore(sess, default_model_path)

    with open('S2VT_prediction.json', 'w') as out:
        generated_sentences = []
        bleu_score_avg = [0., 0.]
        for idx, video in enumerate(test_videos):
            print 'video =>', video

            video_feat_path = os.path.join(video_test_feat_path, video) + '.npy'
            video_feat = np.load(video_feat_path)[None,...]
            #video_feat = np.load(video_feat_path)
            #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            if video_feat.shape[1] == n_frame_step:
                video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
            else:
                continue
                #shape_templete = np.zeros(shape=(1, n_frame_step, 4096), dtype=float )
                #shape_templete[:video_feat.shape[0], :video_feat.shape[1], :video_feat.shape[2]] = video_feat
                #video_feat = shape_templete
                #video_mask = np.ones((video_feat.shape[0], n_frame_step))
    
            generated_word_index = sess.run(caption_tf, feed_dict={video_tf: video_feat, video_mask_tf: video_mask})
            generated_words = ixtoword[generated_word_index]
    
            punctuation = np.argmax(np.array(generated_words) == '<eos>') + 1
            generated_words = generated_words[:punctuation] 
            generated_sentence = ' '.join(generated_words)
            generated_sentence = generated_sentence.replace('<bos> ', '')
            generated_sentence = generated_sentence.replace(' <eos>', '')

            bleu_score = 0.
            print 'generated_sentence =>', generated_sentence
            for reference_sentence in test_labels[idx]['caption']:
                bleu_score += bleu_eval.BLEU_new(generated_sentence, reference_sentence)
            bleu_score_avg[0] += bleu_score
            bleu_score_avg[1] += len(test_labels[idx]['caption'])
            print 'bleu score', bleu_score/len(test_labels[idx]['caption']), '\n'

            generated_sentences.append({"caption": generated_sentence, "id": video})
        print 'avg bleu score', bleu_score_avg[0]/bleu_score_avg[1], '\n'
        json.dump(generated_sentences, out, indent=4)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test(model_path=sys.argv[1])
    else:
        test()
