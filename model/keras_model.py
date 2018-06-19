import numpy as np
from keras.layers import Input, Bidirectional, LSTM, Embedding, Dense, Dropout
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import f1_score

from model.callbacks import F1score

from .data_utils import minibatches, pad_sequences


class BaseKerasModel(object):
    """Generic class for general methods that are not specific to NER"""

    def __init__(self, config):
        """Defines self.config and self.logger

        Args:
            config: (Config instance) class with hyper parameters,
                vocab and embeddings

        """
        self.config = config
        self.logger = config.logger
        self.model = None
        self.sess = None
        self.saver = None

    def batch_iter(self, train, batch_size, return_lengths=False):
        """
        Creates a batch generator for the dataset
        :param train: Dataset
        :param batch_size: Batch Size
        :param return_lengths: If True, generator returns sequence lengths. Used masking data during the evaluation step
        :return: (number of batches in dataset, data generator)
        """
        nbatches = (len(train) + batch_size - 1) // batch_size

        def data_generator():
            while True:
                for i, (words, labels) in enumerate(minibatches(train, batch_size)):

                    # perform padding of the given data
                    if self.config.use_chars:
                        char_ids, word_ids = zip(*words)
                        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
                        char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                        nlevels=2)

                    else:
                        char_ids, word_ids = zip(*words)
                        word_ids, sequence_lengths = pad_sequences(word_ids, 0)

                    if labels:
                        labels, _ = pad_sequences(labels, 0)
                        labels = [to_categorical(label, num_classes=self.config.ntags) for label in labels] # Change labels to one-hot

                    # build dictionary
                    inputs = {
                        "word_ids": np.asarray(word_ids),
                    }

                    if self.config.use_chars:
                        inputs["char_ids"] = np.asarray(char_ids)

                    if return_lengths:
                        yield(inputs, np.asarray(labels), sequence_lengths)

                    else:
                        yield (inputs, np.asarray(labels))

        return (nbatches, data_generator())

    def train(self, train, dev, show_history=False):
        batch_size = self.config.batch_size

        nbatches_train, train_generator = self.batch_iter(train, batch_size)
        nbatches_dev, dev_generator = self.batch_iter(dev, batch_size)


        _, f1_generator = self.batch_iter(dev, batch_size, return_lengths=True)
        f1 = F1score(f1_generator, 2, self.run_evaluate)

        callbacks = self.gen_callbacks([f1])

        history = self.model.fit_generator(generator=train_generator,
                                           steps_per_epoch=2,
                                           validation_data=dev_generator,
                                           validation_steps=2,
                                           epochs=2,
                                           callbacks=callbacks) #, nbatches_train

        if show_history:
            print(history.history['f1'])
            pass


    def predict_words(self, words_raw):
        words = [self.config.processing_word(w) for w in words_raw]
        if type(words[0]) == tuple:
            words = zip(*words)
        char_ids, word_ids = words

        word_ids = np.asarray(word_ids)
        s = word_ids.shape
        word_ids = word_ids.reshape(-1, s[0])
        inputs = [word_ids]

        if self.config.use_chars:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0,
                        nlevels=1)
            char_ids = np.asarray(char_ids)
            s = char_ids.shape
            char_ids = char_ids.reshape(-1, s[0], s[1])
            inputs.append(char_ids)
            #print(word_ids)
            #print(char_ids)

        one_hot_preds = self.model.predict_on_batch(inputs)
        #print("One hot preds: ", one_hot_preds)
        one_hot_preds = [a.flatten() for a in one_hot_preds.squeeze()] #Squeeze to remove unnecessary 1st dimension for batch size
        #print("One hot preds: ", one_hot_preds)

        pred_ids = np.argmax(one_hot_preds, axis=1)
        #print("Pred ids: ", pred_ids)

        preds = [self.idx_to_tag[idx] for idx in pred_ids]

        return preds

    def run_evaluate(self, data_generator, steps_per_epoch):
        accs = []
        label_true = []
        label_pred = []
        for i in range(steps_per_epoch):
            #try:
            x_true, y_true, sequence_lengths = next(data_generator)
            y_pred = self.model.predict_on_batch(x_true)

            for lab, lab_pred, length in zip(y_true, y_pred,
                                             sequence_lengths):
                lab = lab[:length]
                lab_pred = lab_pred[:length]

                lab = np.argmax(lab, axis=1)
                lab_pred = np.argmax(lab_pred, axis=1)
                accs += [a==b for (a, b) in zip(lab, lab_pred)]


                label_true.extend(lab)
                label_pred.extend(lab_pred)


        label_true = np.asarray(label_true)
        #print("Truths: ", label_true)
        label_pred = np.asarray(label_pred)
        #print("Preds: ", label_pred)

        acc = np.mean(accs)

        micro_score = f1_score(label_true, label_pred, average='micro')
        print("acc: ", 100*acc)

        micro_score = f1_score(label_true, label_pred, average='micro')
        print(' - micro f1: {:04.2f}'.format(micro_score * 100))

        macro_score = f1_score(label_true, label_pred, average='macro')
        print(' - macro f1: {:04.2f}'.format(macro_score * 100))

        weighted_score = f1_score(label_true, label_pred, average='weighted')
        print(' - weighted f1: {:04.2f}'.format(weighted_score * 100))

        #print(classification_report(label_true, label_pred))
        return (micro_score, macro_score, weighted_score)

    def get_loss(self):
        return self._loss

    def __getattr__(self, name):
        return getattr(self.model, name)

    def get_optimizer(self):
        return self._optimizer


class Word_BLSTM(BaseKerasModel):
    """
    Basic model for word level Bi-LSTM
    """
    def __init__(self, config):
        super(Word_BLSTM, self).__init__(config)
        self._loss = 'categorical_crossentropy' #losses.sparse_categorical_crossentropy
        self._optimizer = self.config.lr_method # adam
        self.idx_to_tag = {idx: tag for tag, idx in
                           self.config.vocab_tags.items()}
        self.config.use_chars = False

    def build(self):
        input_tensor = Input(shape=(None,), dtype='int32', name='word_ids')
        #masked_input = Masking(mask_value=0)(input_tensor)
        word_embeddings = Embedding(self.config.nwords, self.config.dim_word, mask_zero=True,
                                  weights=[self.config.embeddings], trainable=self.config.train_embeddings)(input_tensor)
        encoded_text = Bidirectional(LSTM(units=self.config.hidden_size_lstm, return_sequences=True))(word_embeddings)
        encoded_text_dropout = Dropout(self.config.dropout)(encoded_text)
        pred = Dense(self.config.ntags, activation='softmax')(encoded_text_dropout)

        self.model = Model(input_tensor, pred)

    def gen_callbacks(self, callbacks_list):
        return callbacks_list




