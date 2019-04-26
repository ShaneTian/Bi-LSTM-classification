from tensorflow import keras


def bi_lstm(vocab_size, embed_size, feature_size, num_classes, regularizers_lambda, dropout_rate):
    inputs = keras.Input(shape=(feature_size,))
    embed = keras.layers.Embedding(vocab_size + 1, embed_size,
                                   embeddings_initializer=keras.initializers.RandomUniform(minval=-1, maxval=1),
                                   mask_zero=True,
                                   input_length=feature_size,
                                   input_shape=(feature_size,))(inputs)
    mask = keras.layers.Masking(mask_value=0)(embed)
    bi_lstm = keras.layers.Bidirectional(keras.layers.LSTM(100,
                                                           bias_initializer=keras.initializers.Constant(0.1),
                                                           dropout=dropout_rate,
                                                           recurrent_dropout=dropout_rate,
                                                           implementation=2))(mask)
    outputs = keras.layers.Dense(num_classes,
                                 activation='softmax',
                                 bias_initializer=keras.initializers.Constant(0.1),
                                 kernel_regularizer=keras.regularizers.l2(regularizers_lambda),
                                 bias_regularizer=keras.regularizers.l2(regularizers_lambda))(bi_lstm)
    model = keras.Model(inputs, outputs)
    return model
