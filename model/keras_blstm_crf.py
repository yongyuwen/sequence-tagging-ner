import math
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Dropout, Lambda, Concatenate
from keras.models import Model
import keras.backend as K
from keras_contrib.layers import CRF
from keras import optimizers
from keras.callbacks import LearningRateScheduler

from .keras_model import BaseKerasModel

class BLSTMCRF(BaseKerasModel):

    def __init__(self, config):
        super(BLSTMCRF, self).__init__(config)
        self._loss = None #losses.sparse_categorical_crossentropy
        self._optimizer = optimizers.Adam(lr=self.config.lr)
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}

    def build(self):
        inputs = [] #Create input for Model

        # build word embeddings
        input_words = Input(shape=(None,), dtype='int32', name='word_ids')
        inputs.append(input_words)
        if self.config.embeddings is None:
            word_embeddings = Embedding(input_dim=self.config.nwords,
                                        output_dim=self.config.dim_word,
                                        mask_zero=True)(input_words)
        else:
            word_embeddings = Embedding(input_dim=self.config.nwords,
                                        output_dim=self.config.dim_word,
                                        mask_zero=True,
                                        weights=[self.config.embeddings],
                                        trainable=self.config.train_embeddings)(input_words)

        # build character based word embedding
        if self.config.use_chars:
            input_chars = Input(batch_shape=(None, None, None), dtype='int32', name='char_ids')
            inputs.append(input_chars)
            char_embeddings = Embedding(input_dim=self.config.nchars,
                                        output_dim=self.config.dim_char,
                                        mask_zero=True,
                                        name='char_embeddings')(input_chars)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config.dim_char)))(char_embeddings)

            # BiLSTM for char_embeddings
            fwd_state = LSTM(self.config.hidden_size_char, return_state=True, name='fw_char_lstm')(char_embeddings)[-2]
            bwd_state = LSTM(self.config.hidden_size_char, return_state=True, go_backwards=True, name='bw_char_lstm')(char_embeddings)[-2]
            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
            # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.config.hidden_size_char]))(char_embeddings)

            #combine characters and word
            word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self.config.dropout)(word_embeddings)
        encoded_text = Bidirectional(LSTM(units=self.config.hidden_size_lstm, return_sequences=True))(word_embeddings)
        encoded_text = Dropout(self.config.dropout)(encoded_text)
        #encoded_text = Dense(100, activation='tanh')(encoded_text)

        if self.config.use_crf:
            crf = CRF(self.config.ntags, sparse_target=False)
            self._loss = crf.loss_function
            pred = crf(encoded_text)

        else:
            self._loss = 'categorical_crossentropy'
            pred = Dense(self.config.ntags, activation='softmax')(encoded_text)

        self.model = Model(inputs, pred)


    def gen_callbacks(self, callbacks_list):
        lrate = LearningRateScheduler(self.step_decay)
        callbacks_list.append(lrate)
        return callbacks_list


    def step_decay(self, epoch):
        initial_lrate = self.config.lr
        decay = self.config.lr_decay
        epochs_drop = self.config.epoch_drop
        lrate = initial_lrate * math.pow(decay,
                                         math.floor((1+epoch)/epochs_drop))
        return lrate

