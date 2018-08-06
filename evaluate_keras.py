from model.data_utils import CoNLLDataset
from model.keras_model import Word_BLSTM
from model.keras_blstm_crf import BLSTMCRF
from model.config import Config


def main():
    # create instance of config
    config = Config()

    model = BLSTMCRF(config) #Word_BLSTM(config)

    model.build()
    model.compile(optimizer=model.get_optimizer(), loss=model.get_loss()) #, metrics=['acc']

    model.load_weights('./saves/test20.h5') #./saves/blstmCrf_15.h5

    test = CoNLLDataset(config.filename_test, config.processing_word,
                        config.processing_tag, config.max_iter)

    batch_size = config.batch_size
    nbatches_test, test_generator = model.batch_iter(test, batch_size, return_lengths=True)

    model.run_evaluate(test_generator, nbatches_test)
    # test predictions
    words = "Fa Mulan is from Dynasty Trading Limited"
    words = words.split(" ")
    pred = model.predict_words(words)
    print(words)
    print(pred)

if __name__ == "__main__":
    main()
