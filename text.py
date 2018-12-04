import numpy as np
import tensorflow as tf

data = open('war_and_peace.txt', 'r').read()
data = list(data)
chars = list(set(data))

VOCAB_SIZE = len(chars)
SEQ_LENGTH = 10
NUM_OF_SEQ = len(data)//SEQ_LENGTH

# Size of tensor containing hidden state of the cell, does not have any correlation with input, output or target
LSTM_SIZE = 256
NUM_EPOCHS = 60

# Data preparation, one-hot encoded characters

ix_to_char = {ix: char for ix, char in enumerate(chars)}
char_to_ix = {char: ix for ix, char in enumerate(chars)}

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
inp = tf.placeholder(tf.float32, shape=(
    None, SEQ_LENGTH, VOCAB_SIZE), name='input')

# A placeholder for targets, i.e. a matrix of shape SEQ_LENGTH x VOCAB_SIZE, the shape of the placeholder is (None, SEQ_LENGTH, VOCAB_SIZE), because None is reserved for batch size
targets = tf.placeholder(tf.float32, shape=(None,
                                            SEQ_LENGTH, VOCAB_SIZE), name='targets')

# Creation of an LSTM cell and setting size of its hidden tensor
cell = tf.contrib.rnn.LSTMBlockCell(num_units=LSTM_SIZE)

# Adding dropout for the cell
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.75)

# Creation of the RNN network with one LSTM cell inside. Input to the network needs to be specified as an argument. The returned values are: outputs and state. Outputs is a list containing a hidden state tensor for every timestep when the network was unrolled. State contains both the last hidden state and cell state, so outputs[-1] should be generally equal to state.h. State will not be used, so its assigned to _
outputs, _ = tf.nn.dynamic_rnn(cell, inp, dtype=tf.float32)

# outputs shape is (?, 10, 256), so the batch size is unknown, 10 is the SEQ_LENGTH and 256 is the LSTM_SIZE - size of the hidden vector flowing inside of the LSTM cell


#  Weights and biases needed for the last layer to prepare the output to be passed to a softmax layer
final_weights = tf.Variable(tf.truncated_normal([LSTM_SIZE, VOCAB_SIZE],
                                                mean=0, stddev=.01))
final_bias = tf.Variable(tf.truncated_normal([VOCAB_SIZE],
                                             mean=0, stddev=.01))

# Get the last timestep, final_output shape is (10, 256)
final_output = outputs[-1]

#  Create logits for the softmax
logits = tf.nn.xw_plus_b(
    final_output, final_weights, final_bias)

#  Pass the output from the network to the softmax activation function and compute the cross entropy after that
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=targets, logits=logits)

# Create an operation for predicting after training - only softmax, not cross entropy, we do not want to calculate loss
prediction = tf.nn.softmax(logits, name='pred')

# Compute mean from loss
cross_entropy = tf.reduce_mean(loss)

# Choose a method for performing backpropagation and tell it to minimize the loss
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

# Output 1 if a character was correctly predicted and 0 if it was not
correct_prediction = tf.equal(tf.argmax(targets, 2),
                              tf.argmax(logits, 1))

# Compute the mean to get the accuracy
accuracy = (tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32)))*100

# Create a session for executing the graph
with tf.Session() as sess:
    # Initialize tf.Variables
    sess.run(tf.global_variables_initializer())

    # Loop over for the number of epochs
    for j in range(NUM_EPOCHS):

        # Feed one SEQ_LENGTH at a time to the network. Since RNN demands batch size, as the first dimension, the list of inputs must be passed to it. Hence, X[i] and y[i] are enclosed in square brackets to make a one-item list from them. Feed_dict is an argument used for passing data to previously defined placeholders. Train_setp and accuracy are computed, and accuracy is printed
        for i in range(0, NUM_OF_SEQ):
            _, acc = sess.run([train_step, accuracy],
                              feed_dict={inp: [X[i]], targets: [y[i]]})

        if (j % 5 == 0 and j != 0):
            print("Accuracy at {} epoch: ".format(j), acc)

    # Testing using prediction operation - every consecutive sentence of length equal to SEQ_LENGTH from the whole text is passed as input, just like during training, but now we expect the network to output the sequence shifted by one character, so ideally generating the same text that was passed to it (excluding the very first character)
    for k in range(NUM_OF_SEQ):
        test_input = X[k]

        out = sess.run(prediction, feed_dict={inp: [test_input]})

        test_output_decoded = [
            ix_to_char[np.argmax(fragment)] for fragment in out]

        print(('').join(test_output_decoded), end='')
