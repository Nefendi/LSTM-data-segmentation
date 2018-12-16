import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)


def create_lstm_layer(num_of_lstms, lstm_size, feedforward_units, input, scope, activation_function, expand_dims):
    cells = [tf.contrib.rnn.LSTMBlockCell(
        num_units=lstm_size, name=f'{scope}_{_}') for _ in range(num_of_lstms)]

    all_outputs = []

    for cell in cells:
        outputs, _ = tf.nn.dynamic_rnn(
            cell, input, dtype=tf.float32)
        all_outputs.append(outputs[-1])

    concatenated_outputs = tf.concat(all_outputs, 1)

    feedforward_outputs = tf.layers.dense(
        inputs=concatenated_outputs, units=feedforward_units, activation=activation_function)

    if expand_dims:
        feedforward_outputs = tf.expand_dims(
            feedforward_outputs, axis=0)

    return feedforward_outputs


with open('war_and_peace.txt', 'r') as f:
    data = f.read()

data = list(data)
chars = list(set(data))

VOCAB_SIZE = len(chars)
SEQ_LENGTH = 10
NUM_OF_SEQ = len(data) // SEQ_LENGTH

# Size of tensor containing hidden state of the cell, does not have any correlation with input, output or target
LSTM_SIZE = 128
# Number of LSTM cells to be used
LSTM_NUMBER_FIRST_LAYER = 3
LSTM_NUMBER_SECOND_LAYER = 2
FIRST_LAYER_UNITS = 64
NUM_EPOCHS = 10

# Data preparation, one-hot encoded characters

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

# Providing the data shifted by one character at a time works worse than shifting by SEQ_LENGTH at a time

# X = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, VOCAB_SIZE))
# y = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, VOCAB_SIZE))
# for i in range(0, NUM_OF_SEQ):
#     X_sequence = data[i:SEQ_LENGTH + i]
#     X_sequence_ix = [char_to_ix[value] for value in X_sequence]
#     input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
#     for j in range(SEQ_LENGTH):
#         input_sequence[j][X_sequence_ix[j]] = 1.
#     X[i] = input_sequence

#     y_sequence = data[i+1:SEQ_LENGTH+(i+1)]
#     y_sequence_ix = [char_to_ix[value] for value in y_sequence]
#     target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
#     for j in range(SEQ_LENGTH):
#         target_sequence[j][y_sequence_ix[j]] = 1.
#     y[i] = target_sequence

X = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, VOCAB_SIZE))
for i in range(0, NUM_OF_SEQ):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1.
    y[i] = target_sequence

######################################################################

# A placeholder for input, i.e. a matrix of shape SEQ_LENGTH x VOCAB_SIZE, the shape of the placeholder is (None, SEQ_LENGTH, VOCAB_SIZE), because None is reserved for batch size
input = tf.placeholder(tf.float32, shape=(
    None, SEQ_LENGTH, VOCAB_SIZE), name='input')

# A placeholder for targets, i.e. a matrix of shape SEQ_LENGTH x VOCAB_SIZE, the shape of the placeholder is (None, SEQ_LENGTH, VOCAB_SIZE), because None is reserved for batch size
targets = tf.placeholder(tf.float32, shape=(
    SEQ_LENGTH, VOCAB_SIZE), name='targets')


first_layer_outputs = create_lstm_layer(num_of_lstms=LSTM_NUMBER_FIRST_LAYER, lstm_size=LSTM_SIZE,
                                        feedforward_units=FIRST_LAYER_UNITS, input=input, scope='lstm_first_layer', activation_function=tf.tanh, expand_dims=True)

logits = create_lstm_layer(num_of_lstms=LSTM_NUMBER_SECOND_LAYER, lstm_size=LSTM_SIZE,
                           feedforward_units=VOCAB_SIZE, input=first_layer_outputs, scope='lstm_second_layer', activation_function=None, expand_dims=False)

#  Pass the output from the network to the softmax activation function and compute the cross entropy after that
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=targets, logits=logits)

# Create an operation for predicting after training - only softmax, not cross entropy, we do not want to calculate loss
prediction = tf.nn.softmax(logits, name='pred')

# Compute mean from loss
cross_entropy = tf.reduce_mean(loss)

# Choose a method for performing backpropagation and tell it to minimize the loss
train_step = tf.train.RMSPropOptimizer(
    learning_rate=0.001, decay=0.9).minimize(cross_entropy)

# Output 1 if a character was correctly predicted and 0 if it was not
correct_prediction = tf.equal(tf.argmax(targets, 1),
                              tf.argmax(logits, 1))

# Compute the mean to get the accuracy
accuracy = (tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32)))*100

print()

# # Create a session for executing the graph
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    # Initialize tf.Variables
    sess.run(tf.global_variables_initializer())

    # Loop over for the number of epochs
    for j in range(NUM_EPOCHS):
        # Feed one SEQ_LENGTH at a time to the network. Since RNN demands batch size, as the first dimension, the list of inputs must be passed to it. Hence, X[i] and y[i] are enclosed in square brackets to make a one-item list from them. Feed_dict is an argument used for passing data to previously defined placeholders. Train_step and accuracy are computed, and accuracy is printed

        total_acc = 0

        for i in range(0, NUM_OF_SEQ):
            _, acc = sess.run([train_step, accuracy],
                              feed_dict={input: [X[i]], targets: y[i]})
            total_acc += acc

        print(f"Accuracy at epoch {j+1}: {(total_acc / NUM_OF_SEQ):.2f}")

# Testing using prediction operation - every consecutive sentence of length equal to SEQ_LENGTH from the whole text is passed as input, just like during training, but now we expect the network to output the sequence shifted by one character, so ideally generating the same text that was passed to it (excluding the very first character)
# for k in range(NUM_OF_SEQ):
#     test_input = X[k]

#     out = sess.run(prediction, feed_dict={input: [test_input]})

#     if (k % SEQ_LENGTH == 0):
#         test_output_decoded = [
#             ix_to_char[np.argmax(fragment)] for fragment in out]
#         print(('').join(test_output_decoded), end='')

    print()

    for k in range(NUM_OF_SEQ):
        test_input = X[k]

        out = sess.run(prediction, feed_dict={input: [test_input]})

        test_output_decoded = [
            ix_to_char[np.argmax(fragment)] for fragment in out]

        print(('').join(test_output_decoded), end='')
