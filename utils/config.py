class Config():
    train = '../../../data/ctb5/train.conll'
    dev = '../../../data/ctb5/dev.conll'
    test = '../../../data/ctb5/test.conll'
    embed = '../../../data/giga.100.txt'


class CHAR_LSTM_CRF_Config(Config):
    context_size = 1
    embedding_dim = 100
    char_embedding_dim = 100
    char_output_size = 200
    hidden_size = 150
    use_char = True


config = {'char_lstm_crf': CHAR_LSTM_CRF_Config}