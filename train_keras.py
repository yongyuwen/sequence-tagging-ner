from model.data_utils import CoNLLDataset
from model.keras_model import Word_BLSTM
from model.keras_blstm_crf import BLSTMCRF
from model.config import Config
from keras.models import load_model



def main():
    # create instance of config
    config = Config()

    #build model
    model = BLSTMCRF(config) #Word_BLSTM(config)
    #model = Word_BLSTM(config)
    model.build()
    model.compile(optimizer=model.get_optimizer(), loss=model.get_loss()) #, metrics=['acc']

    #model.summary()
    # Loading weights
    #model.load_weights('./saves/test20.h5')


    # create datasets
    dev = CoNLLDataset(config.filename_train, config.processing_word, #filename_dev
                         config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    model.summary()
    # train model
    model.train(train, dev)

    # Save model
    model.save_weights('./saves/test20.h5')

    # test predictions
    # words = "Obama was born in hawaii"
    # words = words.split(" ")
    # pred = model.predict_words(words)
    # print(pred)

if __name__ == "__main__":
    main()
