import numpy as np
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Dropout, Lambda, Concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
from keras_contrib.layers import crf


from .callbacks import F1score
from .keras_model import BaseKerasModel
from .data_utils import minibatches, pad_sequences

class BLSTMCRF(BaseKerasModel):
    def __init__(self, config):
        super(BLSTMCRF, self).__init__(config)
        self._loss = 'categorical_crossentropy' #losses.sparse_categorical_crossentropy
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
                                        mask_zero=True)(input_chars)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.config.dim_char)))(char_embeddings)

            # BiLSTM for char_embeddings
            fwd_state = LSTM(self.config.hidden_size_char, return_state=True)(char_embeddings)[-2]
            bwd_state = LSTM(self.config.hidden_size_char, return_state=True, go_backwards=True)(char_embeddings)[-2]
            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
            # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.config.hidden_size_char]))(char_embeddings)

            #combine characters and word
            word_embeddings = Concatenate(axis=-1)([word_embeddings, char_embeddings])

        word_embeddings = Dropout(self.config.dropout)(word_embeddings)
        encoded_text = Bidirectional(LSTM(units=self.config.hidden_size_lstm, return_sequences=True))(word_embeddings)
        encoded_text = Dropout(self.config.dropout)(encoded_text)
        pred = Dense(self.config.ntags, activation='softmax')(encoded_text)

        self.model = Model(inputs, pred)


    # def batch_iter(self, train, batch_size, return_lengths=False):
    #     """
    #     Creates a batch generator for the dataset
    #     :param train: Dataset
    #     :param batch_size: Batch Size
    #     :param return_lengths: If True, generator returns sequence lengths. Used masking data during the evaluation step
    #     :return: (number of batches in dataset, data generator)
    #     """
    #     nbatches = (len(train) + batch_size - 1) // batch_size
    #
    #     def data_generator():
    #         while True:
    #             for i, (words, labels) in enumerate(minibatches(train, batch_size)):
    #
    #                 # perform padding of the given data
    #                 if self.config.use_chars:
    #                     char_ids, word_ids = zip(*words)
    #                     word_ids, sequence_lengths = pad_sequences(word_ids, 0)
    #                     char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
    #                     nlevels=2)
    #
    #                 else:
    #                     char_ids, word_ids = zip(*words)
    #                        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
    #
    #                 if labels:
    #                     labels, _ = pad_sequences(labels, 0)
    #                     labels = [to_categorical(label, num_classes=self.config.ntags) for label in labels] # Change labels to one-hot
    #
    #                 # build dictionary
    #                 inputs = {
    #                     "word_ids": np.asarray(word_ids),
    #                 }
    #
    #                 if self.config.use_chars:
    #                     inputs["char_ids"] = np.asarray(char_ids)
    #
    #                 if return_lengths:
    #                     yield(inputs, np.asarray(labels), sequence_lengths)
    #
    #                 else:
    #                     yield (inputs, np.asarray(labels))
    #
    #     return (nbatches, data_generator())

