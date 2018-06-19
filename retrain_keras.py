from model.keras_blstm_crf import BLSTMCRF
from model.config import Config
from model.data_utils import CoNLLDataset, get_vocabs, UNK, NUM, \
    get_glove_vocab, write_vocab, load_vocab, get_char_vocab, \
    export_trimmed_glove_vectors, get_processing_word
import numpy as np
import os

def main():
    # create instance of config
    config = Config()
    #


    # 1. Load previous vocab words
    old_vocab = set()
    with open(config.filename_words) as f:
        for word in f:
            #print(word)
            old_vocab.add(word.strip())
    print("Number of old vocabs = ", len(old_vocab))

    # Load new vocab and check for words in new vocab that is not in old vocab
    processing_word = get_processing_word(lowercase=True)
    dev   = CoNLLDataset(config.filename_dev, processing_word)
    test  = CoNLLDataset(config.filename_test, processing_word)

    vocab_words, vocab_tags = get_vocabs([dev, test])

    # Get vocab in new dataset that is not in old vocab
    vocab_new = vocab_words - old_vocab
    print("Number of new words: ", len(vocab_new))

    # Get full glove vocab
    vocab_glove = get_glove_vocab(config.filename_glove)

    # Get vocab set for words in new vocab and in glove_vocab
    vocab = vocab_new & vocab_glove
    print("Final number of additions are: ", len(vocab))

    # Load old model
    model = BLSTMCRF(config)
    model.build()
    model.summary()
    model.load_weights('./saves/less_words.h5')
    embedding_weights = model.get_layer(name="word_embeddings").get_weights()[0]
    print(embedding_weights.shape)


    def create_embedding_dict(glove_dir, dim_size):
        print("Creating embedding dictionary...")
        embeddings_index = {}
        f = open(glove_dir, encoding='utf-8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        print('Found %s word vectors.' % len(embeddings_index))
        return embeddings_index

    embeddings_index = create_embedding_dict(config.filename_glove, config.dim_word)

    # Create new embedding size
    embeddings = np.zeros([embedding_weights.shape[0]+len(vocab), embedding_weights.shape[1]])
    # Load old vectors
    for idx, vec in enumerate(embedding_weights):
        embedding_weights[idx] = vec
    # Load new vectors
    pt = embedding_weights.shape[0]
    for idx, word in enumerate(vocab):
        embeddings[idx+pt] = embeddings_index.get(word)
    print("Size of new embeddings: ", embeddings.shape)
    # Save embeddings to npz
    np.savez_compressed(config.filename_trimmed, embeddings=embeddings)

    # Write new vocab file for new config
    def append_vocab(vocab, filename):
        """Writes a vocab to a file

        Writes one word per line.

        Args:
            vocab: iterable that yields word
            filename: path to vocab file

        Returns:
            write a word per line

        """
        print("Writing vocab...")
        with open(filename, "a") as f:
            f.write("\n")
            for i, word in enumerate(vocab):
                if i != len(vocab) - 1:
                    f.write("{}\n".format(word))
                else:
                    f.write(word)
        print("- done. {} tokens".format(len(vocab)))
    append_vocab(vocab, config.filename_words)

    # Build new model
    config2 = Config()
    model2 = BLSTMCRF(config2)
    model2.build()
    model2.summary()

    layer_names = ["char_embeddings", "fw_char_lstm", "bw_char_lstm", "bidirectional", "crf"]

    # Set other weights
    for layer_name in layer_names:
        if layer_name == "crf":
            model2.get_layer(name="crf_2").set_weights(model.get_layer(name="crf_1").get_weights())
        else:
            model2.get_layer(name=layer_name).set_weights(model.get_layer(name=layer_name).get_weights())

    # Set embedding weights
    #model2.get_layer(name="word_embeddings").set_weights([embeddings])
    model2.summary()
    model2.save_weights('./saves/WEWWWWW.h5')

if __name__ == "__main__":
    main()
