import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
import math

class Video_Caption_Generator():
    def __init__(self, batch_size, n_words, dim_hidden, dim_image, n_video_lstm_step, n_caption_lstm_step, bias_init_vector=None):
        self.batch_size = batch_size
        self.n_words = n_words
        self.dim_hidden = dim_hidden
        self.dim_image = dim_image
        self.n_video_lstm_step = n_video_lstm_step
        self.n_caption_lstm_step = n_caption_lstm_step
        self.bias_init_vector = bias_init_vector

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.truncated_normal([n_words, dim_hidden], stddev=6/math.sqrt(dim_hidden)), name='Wemb')

        self.embed_word_W = tf.Variable(tf.truncated_normal([dim_hidden, n_words], -0.1, 0.1), name='embed_word_W')
        if bias_init_vector is not None:
            self.embed_word_b = tf.Variable(bias_init_vector, dtype=tf.float32, name='embed_word_b')
        else:
            self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')

        self.lstm1 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)
        self.lstm2 = tf.contrib.rnn.BasicLSTMCell(dim_hidden, state_is_tuple=True)

        self.encode_image_W = tf.Variable(tf.truncated_normal([dim_image, dim_hidden], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([dim_hidden]), name='encode_image_b')

        self.attention_W = tf.Variable(tf.truncated_normal([n_video_lstm_step, n_video_lstm_step], -0.01, 0.01), name='attention_W')
        # self.attention_X = tf.Variable(tf.zeros([1, batch_size, dim_hidden]), name='attention_X')
        self.attention_b = tf.Variable(tf.zeros([n_video_lstm_step]), name='attention_b')

    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image]) # b x n x d
        video_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step])
        video_flat = tf.reshape(video, [-1, self.dim_image]) # (b x n) x d

        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b) # (b x n) x d
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.dim_hidden]) # b x n x d

        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        state1 = self.lstm1.zero_state(self.batch_size, dtype=tf.float32)
        state2 = self.lstm2.zero_state(self.batch_size, dtype=tf.float32)
        padding = tf.zeros([self.batch_size, self.dim_hidden])
        # attention_padding = tf.zeros([self.batch_size, self.dim_hidden])

        probs = []
        loss = 0.0

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(tf.concat([padding, image_emb[:, i, :]], 1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
                output2 = tf.reshape(output2, [self.batch_size, self.dim_hidden, 1])
                if i == 0:
                    attention_X = output2
                else:
                    attention_X = tf.concat([attention_X, output2], 2) # b x h x n

        attention_X = tf.reshape(attention_X, [-1, self.n_video_lstm_step]) # (b x h) x n
        attention = tf.nn.xw_plus_b(attention_X, self.attention_W, self.attention_b) # (b x h) x n
        attention = tf.reshape(attention, [self.batch_size, self.dim_hidden, self.n_video_lstm_step]) # b x h x n
        attention = tf.reduce_sum(attention, 2) # b x h

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step):
            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:, i])

            tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(tf.concat([attention, padding], 1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            labels = tf.expand_dims(caption[:, i+1], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat([indices, labels], 1)
            onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
            cross_entropy = cross_entropy * caption_mask[:,i]
            probs.append(logit_words)

            current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
            loss = loss + current_loss

        return loss, video, video_mask, caption, caption_mask, probs

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])
        video_mask = tf.placeholder(tf.float32, [1, self.n_video_lstm_step])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.dim_hidden])

        state1 = self.lstm1.zero_state(1, dtype=tf.float32)
        state2 = self.lstm2.zero_state(1, dtype=tf.float32)
        padding = tf.zeros([1, self.dim_hidden])
        # attention_padding = tf.zeros([1, self.dim_hidden])

        video_caption = []

        probs = []
        embeds = []
        
        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(tf.concat([padding, image_emb[:, i, :]], 1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
                output2 = tf.reshape(output2, [1, self.dim_hidden, 1])
                if i == 0:
                    attention_X = output2
                else:
                    attention_X = tf.concat([attention_X, output2], 2) # 1 x h x n

        attention_X = tf.reshape(attention_X, [-1, self.n_video_lstm_step]) # (1 x h) x n
        attention = tf.nn.xw_plus_b(attention_X, self.attention_W, self.attention_b) # (1 x h) x n
        attention = tf.reshape(attention, [1, self.dim_hidden, self.n_video_lstm_step]) # 1 x h x n
        attention = tf.reduce_sum(attention, 2) # 1 x h

        ############################# Decoding Stage ######################################
        for i in range(0, self.n_caption_lstm_step):
            tf.get_variable_scope().reuse_variables()

            if i == 0:
                with tf.device('/cpu:0'):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(tf.concat([attention, padding], 1), state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            max_prob_index = tf.argmax(logit_words, 1)[0]
            video_caption.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, video_mask, video_caption, probs, embeds
