"""
The hyperparameters used in the model:
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
import os
from builtins import map
from collections import Counter

flags = tf.flags

flags.DEFINE_string("op", None, "Operation: train or test.")
flags.DEFINE_string("id_path", None, "Where the id is stored.")
flags.DEFINE_string("feat_path", None, "Where the feature is stored.")
flags.DEFINE_string("cap_path", None, "(training) Where the caption is stored.")
flags.DEFINE_string("save_path", None, "Where to save/load the model.")
flags.DEFINE_string("gen_path", None, "(testing) Where to output caption generation results.")
FLAGS = flags.FLAGS

#--------------- Fixed Parameters --------------#
dim_image = 4096
n_frame_step = 80
#-------------- Tunable Parameters --------------#
dim_hidden= 256
n_caption_step = 41
n_epochs = 700
vocab_size = 5000
batch_size = 200
learning_rate = 0.001
#---------------------------------------------- #


class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, dim_hidden, batch_size, n_lstm_steps, drop_out_rate, bias_init_vector=None):
        self.dim_image = dim_image
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.batch_size = batch_size
        self.n_lstm_steps = n_lstm_steps
        self.drop_out_rate = drop_out_rate

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_hidden], -0.1, 0.1), name='Wemb')

        self.lstm3 = tf.contrib.rnn.LSTMCell(self.dim_hidden,2*self.dim_hidden,
            use_peepholes = True, state_is_tuple = False)
        self.lstm3_dropout = tf.contrib.rnn.DropoutWrapper(self.lstm3,output_keep_prob=1 - self.drop_out_rate)

        self.encode_image_W = tf.Variable( tf.random_uniform([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable( tf.zeros([dim_hidden]), name='encode_image_b')
        self.embed_att_w = tf.Variable(tf.random_uniform([dim_hidden, 1], -0.1,0.1), name='embed_att_w')
        self.embed_att_Wa = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden], -0.1,0.1), name='embed_att_Wa')
        self.embed_att_Ua = tf.Variable(tf.random_uniform([dim_hidden, dim_hidden],-0.1,0.1), name='embed_att_Ua')
        self.embed_att_ba = tf.Variable( tf.zeros([dim_hidden]), name='embed_att_ba')

        self.embed_word_W = tf.Variable(tf.random_uniform([dim_hidden, n_words], -0.1,0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.embed_nn_Wp = tf.Variable(tf.random_uniform([3*dim_hidden, dim_hidden], -0.1,0.1), name='embed_nn_Wp')
        self.embed_nn_bp = tf.Variable(tf.zeros([dim_hidden]), name='embed_nn_bp')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image]) # b x n x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps]) # b x n

        caption = tf.placeholder(tf.int32, [self.batch_size, n_caption_step]) # b x 16
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, n_caption_step]) # b x 16

        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x h
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden]) # b x n x h
        image_emb = tf.transpose(image_emb, [1,0,2]) # n x b x h

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size]) # b x s
        h_prev = tf.zeros([self.batch_size, self.dim_hidden]) # b x h

        loss_caption = 0.0

        current_embed = tf.zeros([self.batch_size, self.dim_hidden]) # b x h
        #brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_lstm_steps,1,1]) # n x h x 1
        #image_part = tf.batch_matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_lstm_steps,1,1])) + self.embed_att_ba # n x b x h
        image_part = tf.reshape(image_emb, [-1, self.dim_hidden])
        image_part = tf.matmul(image_part, self.embed_att_Ua) + self.embed_att_ba
        image_part = tf.reshape(image_part, [self.n_lstm_steps, self.batch_size, self.dim_hidden])
        with tf.variable_scope("model") as scope:
            for i in range(n_caption_step):
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
                #e = tf.batch_matmul(e, brcst_w)    # unnormalized relevance score 
                e = tf.reshape(e, [-1, self.dim_hidden])
                e = tf.matmul(e, self.embed_att_w) # n x b
                e = tf.reshape(e, [self.n_lstm_steps, self.batch_size])
                #e = tf.reduce_sum(e,2) # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b 
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))   # regularize denominator
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h  # normalize to obtain alpha
                attention_list = tf.multiply(alphas, image_emb) # n x b x h
                atten = tf.reduce_sum(attention_list,0) # b x h       #  soft-attention weighted sum
                #if i > 0: tf.get_variable_scope().reuse_variables()
                if i > 0: scope.reuse_variables()

                with tf.variable_scope("LSTM3"):
                    output1, state1 = self.lstm3_dropout(tf.concat([atten, current_embed], 1), state1 ) # b x h

                output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed], 1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
                h_prev = output1 # b x h
                labels = tf.expand_dims(caption[:,i], 1) # b x 1
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1) # b x 1
                concated = tf.concat([indices, labels], 1) # b x 2
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0) # b x w
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i])

                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) # b x w
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logit_words, labels = onehot_labels) # b x 1
                cross_entropy = cross_entropy * caption_mask[:,i] # b x 1
                loss_caption += tf.reduce_sum(cross_entropy) # 1

        loss_caption = loss_caption / tf.reduce_sum(caption_mask)
        loss = loss_caption
        return loss, video, video_mask, caption, caption_mask


    def build_generator(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_lstm_steps])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b( video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_lstm_steps, self.dim_hidden])
        image_emb = tf.transpose(image_emb, [1,0,2])

        state1 = tf.zeros([self.batch_size, self.lstm3.state_size])
        h_prev = tf.zeros([self.batch_size, self.dim_hidden])

        generated_words = []

        current_embed = tf.zeros([self.batch_size, self.dim_hidden])
        brcst_w = tf.tile(tf.expand_dims(self.embed_att_w, 0), [self.n_lstm_steps,1,1])   # n x h x 1
        #image_part = tf.batch_matmul(image_emb, tf.tile(tf.expand_dims(self.embed_att_Ua, 0), [self.n_lstm_steps,1,1])) +  self.embed_att_ba # n x b x h
        image_part = tf.reshape(image_emb, [-1, self.dim_hidden])
        image_part = tf.matmul(image_part, self.embed_att_Ua) + self.embed_att_ba
        image_part = tf.reshape(image_part, [self.n_lstm_steps, self.batch_size, self.dim_hidden])
        with tf.variable_scope("model") as scope:
            #scope.reuse_variables()
            for i in range(n_caption_step):
                e = tf.tanh(tf.matmul(h_prev, self.embed_att_Wa) + image_part) # n x b x h
                #e = tf.batch_matmul(e, brcst_w)
                e = tf.reshape(e, [-1, self.dim_hidden])
                e = tf.matmul(e, self.embed_att_w) # n x b
                e = tf.reshape(e, [self.n_lstm_steps, self.batch_size])
                #e = tf.reduce_sum(e,2) # n x b
                e_hat_exp = tf.multiply(tf.transpose(video_mask), tf.exp(e)) # n x b
                denomin = tf.reduce_sum(e_hat_exp,0) # b
                denomin = denomin + tf.to_float(tf.equal(denomin, 0))
                alphas = tf.tile(tf.expand_dims(tf.div(e_hat_exp,denomin),2),[1,1,self.dim_hidden]) # n x b x h
                attention_list = tf.multiply(alphas, image_emb) # n x b x h
                atten = tf.reduce_sum(attention_list,0) # b x h

                if i > 0: scope.reuse_variables()

                with tf.variable_scope("LSTM3") as vs:
                    output1, state1 = self.lstm3( tf.concat([atten, current_embed], 1), state1 ) # b x h
                    lstm3_variables = [v for v in tf.global_variables() if v.name.startswith(vs.name)]

                output2 = tf.tanh(tf.nn.xw_plus_b(tf.concat([output1,atten,current_embed], 1), self.embed_nn_Wp, self.embed_nn_bp)) # b x h
                h_prev = output1
                logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b) # b x w
                max_prob_index = tf.argmax(logit_words, 1) # b
                generated_words.append(max_prob_index) # b
                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)

        generated_words = tf.transpose(tf.stack(generated_words))
        return video, video_mask, generated_words, lstm3_variables


def get_video_id(video_id_path):
    with open(video_id_path, "r") as f:
        video_ids = f.readlines()
        video_ids = [x.strip("\n") for x in video_ids] 
    return video_ids

def get_video_feat(video_ids, video_feat_path):
    feats = np.zeros((len(video_ids), n_frame_step, dim_image))
    for i in range(len(video_ids)):
        feats[i, :, :] = np.load(video_feat_path + video_ids[i] + ".npy")
    return feats

# captions is a list of lists with len equal to the number of videos, and each list element is also a
# list with viable captions for its video content
def get_video_caption(video_ids, video_caption_path):
    with open(video_caption_path) as f:
        data = json.load(f)
    captions = [d["caption"] for d in data]
    return captions

def preprocess_caption(captions):
    for i in range(len(captions)):
        for j in range(len(captions[i])):
            captions[i][j] = captions[i][j].replace('.', '')
            captions[i][j] = captions[i][j].replace(',', '')
    return captions

def build_vocab(sentence_iterator, vocab_size): # NeuralTalk
    sentence_iterator = [sent for sub_list in sentence_iterator for sent in sub_list ]
    word_counts = Counter()
    nsents = 0
    for sent in sentence_iterator:
        nsents += 1
        for w in sent.lower().split(' '):
            word_counts[w] += 1

    # Storing only the most common words based on vocab_size
    vocab = word_counts.most_common(vocab_size)
    # Decouple the tuples in vocab
    vocab = [word[0] for word in vocab]

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector

# Pads list of lists into list of equal lengthed lists with trailing zeros
# and then further transform it into numpy matrix with dimension: (list_num, length)
def zero_padding(list_of_lists, length):
    padded_list = []
    for sub_list in list_of_lists:
        if length < len(sub_list):
            raise ValueError("zero padding's max length ", length, 
                "is smaller than a sentence with length =" , len(sub_list),"!")
        padded_list.append(sub_list + [0] * (length - len(sub_list)))
    return np.array(padded_list)

# This function generates a list of index, based on the length of each captions
# the purpose of this function is to iterate over all possible descriptions for
# each videos. , the index is cyclic shifted after each epoch.
def update_caption_iterator(caption_len_list, iter_list, epoch):
    if epoch < 500:
        for i in range(len(caption_len_list)):
            if iter_list[i] < (caption_len_list[i]-1):
                iter_list[i] += 1
            else:
                iter_list[i] = 0

    return iter_list

def train():
    if os.path.exists(FLAGS.save_path):
        raise ValueError(FLAGS.save_path + "exists! Exiting...")
    else:
        os.makedirs(FLAGS.save_path)
    ids = get_video_id(FLAGS.id_path)
    feats = get_video_feat(ids, FLAGS.feat_path)
    captions = get_video_caption(ids, FLAGS.cap_path)
    captions = preprocess_caption(captions)
    caption_len_list = [len(sub_list) for sub_list in captions]
    num_training_data = len(captions)
    
    # initialize the caption iterator to all zeros with length equal to number of videos 
    caption_iterator = [0]*len(captions)

    wordtoix, ixtoword, bias_init_vector = build_vocab(captions, vocab_size)

    if not os.path.exists("data/"):
        os.makedirs("data/")
    np.save('data/ixtoword', ixtoword)
    
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoix),
            dim_hidden=dim_hidden,
            batch_size=batch_size,
            n_lstm_steps=n_frame_step,
            drop_out_rate=0.5,
            bias_init_vector=bias_init_vector)

    tf_loss, tf_video, tf_video_mask, tf_caption, tf_caption_mask = model.build_model()
    sess = tf.InteractiveSession()

    saver = tf.train.Saver(max_to_keep=2)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)
    sess.run(tf.global_variables_initializer())
    index = list(range(num_training_data))
    for epoch in range(n_epochs):
        # Shuffle training data based on randomly generated index
        np.random.shuffle(index)
        feats = feats[index, :, :]
        captions = [captions[i] for i in index]
        caption_len_list = [caption_len_list[i] for i in index]
        caption_iterator = [caption_iterator[i] for i in index]

        loss_epoch = []
        for start,end in zip(
                range(0, num_training_data, batch_size),
                range(batch_size, num_training_data, batch_size)):

            current_feats = feats[start:end, :, :]
            current_video_masks = np.zeros((batch_size, n_frame_step))

            for ind,feat in enumerate(current_feats):
                current_video_masks[ind][:len(current_feats[ind])] = 1

            current_captions = [captions[i][caption_iterator[i]] for i in range(start,end)]

            current_caption_ind = list(map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ') if word in wordtoix], current_captions))

            current_caption_matrix = zero_padding(current_caption_ind, length=n_caption_step-1)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix),1]) ] ).astype(int)
            current_caption_masks = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( list(map(lambda x: (x != 0).sum()+1, current_caption_matrix )))

            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1

            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_feats,
                        tf_video_mask : current_video_masks,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })

            loss_epoch.append(loss_val)
        # After each epoch, cyclic shift the caption iterator
        caption_iterator = update_caption_iterator(caption_len_list, caption_iterator, epoch)

        print("Epoch:", epoch, "\t Loss:", np.mean(loss_epoch))
        if epoch == n_epochs-1:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, FLAGS.save_path, global_step=epoch)

def test():
    ids = get_video_id(FLAGS.id_path)
    feats = get_video_feat(ids, FLAGS.feat_path)
    ixtoword = np.load('./data/ixtoword.npy').tolist()
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(ixtoword),
            dim_hidden=dim_hidden,
            batch_size=1,
            n_lstm_steps=n_frame_step,
            drop_out_rate = 0,
            bias_init_vector=None)

    video_tf, video_mask_tf, caption_tf, lstm3_variables_tf = model.build_generator()
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    
    with tf.device("/cpu:0"):
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.save_path)
    for ind, row in enumerate(lstm3_variables_tf):
        if ind % 4 == 0:
            assign_op = row.assign(tf.multiply(row,1-0.5))
            sess.run(assign_op)

    save_list = []
    for i in range(len(ids)):
        idx = ids[i]
        video_feat = feats[i,:,:].reshape((1,80,4096))
        video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))

        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat, video_mask_tf:video_mask})
        #probs_val = sess.run(probs_tf, feed_dict={video_tf:video_feat})
        #embed_val = sess.run(last_embed_tf, feed_dict={video_tf:video_feat})

        generated_words = [ixtoword[generated_word_index[0,i]] for i in range(generated_word_index.shape[1])]
        punctuation = np.argmax(np.array(generated_words) == '.')# + 1 to include period
        generated_words = generated_words[:punctuation]
        generated_sentence = ' '.join(generated_words)

        save_list.append({"caption": generated_sentence, "id": idx})

    with open(FLAGS.gen_path, 'w') as f:
        json.dump(save_list, f, indent=2)


def main(_):
    if FLAGS.op not in ["train", "test"]:
        raise ValueError("Must set --op to either train or test")
    if not FLAGS.id_path:
        raise ValueError("Must set --id_path to id of videos")
    if not FLAGS.feat_path:
        raise ValueError("Must set --feat_path to the features of the videos in id_path")
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path to the directory of your model")

    if FLAGS.op == "train":
        if not FLAGS.cap_path:
            raise ValueError("Must set --cap_path to the directory of training data's caption")
        train()
    
    elif FLAGS.op == "test":
        if not FLAGS.gen_path:
            raise ValueError("Must set --gen_path to the path for generation output")
        test()
    else:
        raise ValueError("I don't know how you got here...")


if __name__ == "__main__":
    tf.app.run()
