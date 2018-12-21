import numpy as np
import tensorflow as tf
from textblob import TextBlob
from random import shuffle

tf.logging.set_verbosity(tf.logging.ERROR)


def create_lstm_layer(num_of_lstms, lstm_size, feedforward_units, input, scope, activation_function, expand_dims):
    # Create a pool of LSTM cells
    cells = [tf.contrib.rnn.LSTMBlockCell(
        num_units=lstm_size, name=f'{scope}_{_}') for _ in range(num_of_lstms)]

    all_outputs = []

    # Unroll each LSTM for each entry in the input sequence and aggregate them
    for cell in cells:
        outputs, _ = tf.nn.dynamic_rnn(
            cell, input, dtype=tf.float32)
        all_outputs.append(outputs[-1])

    # Concatenate outputs so that each entry in the input sequence has answers from each LSTM cell
    concatenated_outputs = tf.concat(all_outputs, 1)

    # Pass the concatenated outputs to a feedforward layer to rescale the shapes of the tensors
    feedforward_outputs = tf.layers.dense(
        inputs=concatenated_outputs, units=feedforward_units, activation=activation_function)

    # If this is not the output layer, expand dimensions, so that the outputs can be passed to the next LSTM layer (TensorFlow syntax needs this)
    if expand_dims:
        feedforward_outputs = tf.expand_dims(
            feedforward_outputs, axis=0)

    return feedforward_outputs


def create_lstm_layer_with_backward_pass(num_of_fw_lstms, num_of_bw_lstms, lstm_size, feedforward_units, input, scope, activation_function, expand_dims):
     # Create a pool of forward LSTM cells
    fw_cells = [tf.contrib.rnn.LSTMBlockCell(
        num_units=lstm_size, name=f'{scope}_fw_{_}') for _ in range(num_of_fw_lstms)]

    # Create a pool of backward LSTM cells
    bw_cells = [tf.contrib.rnn.LSTMBlockCell(
        num_units=lstm_size, name=f'{scope}_bw_{_}') for _ in range(num_of_bw_lstms)]

    all_outputs = []

    # Unroll each forward LSTM for each entry in the input sequence and aggregate them
    for fw_cell in fw_cells:
        outputs, _ = tf.nn.dynamic_rnn(
            fw_cell, input, dtype=tf.float32)
        all_outputs.append(outputs[-1])

    # Unroll each backward LSTM for each entry in the input sequence, reverse the input sequence and then reverse the outputs and aggregate them
    for bw_cell in bw_cells:
        outputs, _ = tf.nn.dynamic_rnn(
            bw_cell, tf.reverse(input, [1]), dtype=tf.float32)
        all_outputs.append(tf.reverse(outputs[-1], [1]))

    # Concatenate outputs so that each entry in the input sequence has answers from each LSTM cell
    concatenated_outputs = tf.concat(all_outputs, 1)

    # Pass the concatenated outputs to a feedforward layer to rescale the shapes of the tensors
    feedforward_outputs = tf.layers.dense(
        inputs=concatenated_outputs, units=feedforward_units, activation=activation_function)

    # If this is not the output layer, expand dimensions, so that the outputs can be passed to the next LSTM layer (TensorFlow syntax needs this)
    if expand_dims:
        feedforward_outputs = tf.expand_dims(
            feedforward_outputs, axis=0)

    return feedforward_outputs


with open('war_and_peace.txt', 'r') as f:
    data = f.read()

# Create a TextBlob object from the input data
data = TextBlob(data)
# Get a part-of-speech tag describing each word and strip the text of all additional characters like commas, dots, hyphens, etc.
words_pos_tags = data.tags
# Obtain words from tuples of (word, pos_tag) in order to encode them
words = [word for word, tag in words_pos_tags]
# Remove duplicates
unique_words = list(set(words))

words_pos_tags_indexed = []

# Turn POS tags into numbers in the following way:
# - noun --> 0
# - verb --> 1
# - adjective --> 2
# - adverb --> 3
# - other --> 4
for word, tag in words_pos_tags:
    if (tag[0] == 'N'):
        words_pos_tags_indexed.append((word, 0))
    elif (tag[0] == 'V'):
        words_pos_tags_indexed.append((word, 1))
    elif (tag[0] == 'J'):
        words_pos_tags_indexed.append((word, 2))
    elif (tag[0] == 'R'):
        words_pos_tags_indexed.append((word, 3))
    else:
        words_pos_tags_indexed.append((word, 4))

# A dictionary to turn an index into a word
ix_to_word = {ix: word for ix, word in enumerate(unique_words)}
# A dictionary to turn a word into an index
word_to_ix = {word: ix for ix, word in enumerate(unique_words)}
# A dictionary to turn an index into a part of speech
ix_to_tag = {0: 'NOUN', 1: 'VERB', 2: 'ADJECTIVE', 3: 'ADVERB', 4: 'OTHER'}

# Number of unique words
VOCAB_SIZE = len(unique_words)
# Number of labels
NUM_OF_CLASSES = len(ix_to_tag)
# Length of one sequence of words input to the network
SEQ_LENGTH = 10
# Number of sequences that can be created from the input data
NUM_OF_SEQ = len(words) // SEQ_LENGTH
# How much of the data is to be used for training
TRAIN_DATA_FRACTION = 7 / 8
# Set the number of training and test sequences
TRAIN_DATA_NUM = int(TRAIN_DATA_FRACTION * NUM_OF_SEQ)
TEST_DATA_NUM = NUM_OF_SEQ - TRAIN_DATA_NUM

OUTPUT_FILE = "results.txt"

# Size of tensor containing hidden state of the cell, does not have any correlation with input, output or target
LSTM_SIZE = 128
# Number of LSTM cells to be used in the first layer
LSTM_FW_NUMBER_FIRST_LAYER = 2
LSTM_BW_NUMBER_FIRST_LAYER = 2
LSTM_ONLY_FW_NUMBER_FIRST_LAYER = 3

# Number of LSTM cells to be used in the second layer
LSTM_FW_NUMBER_SECOND_LAYER = 1
LSTM_BW_NUMBER_SECOND_LAYER = 1
LSTM_ONLY_FW_NUMBER_SECOND_LAYER = 2

# Number of output units from the feedforward layer in the first layer
FIRST_LAYER_UNITS = 64
# Number of epochs to be spend on learning
NUM_EPOCHS = 20

# Data preparation, one-hot encoded words and labels
X = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((NUM_OF_SEQ, SEQ_LENGTH, NUM_OF_CLASSES))
for i in range(0, NUM_OF_SEQ):
    X_sequence = words_pos_tags_indexed[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [word_to_ix[word] for word, tag in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
    X[i] = input_sequence

    target_sequence = np.zeros((SEQ_LENGTH, NUM_OF_CLASSES))
    for j in range(SEQ_LENGTH):
        target_sequence[j][X_sequence[j][1]] = 1.
    y[i] = target_sequence

# Divide the dataset into training and testing parts
trainX = X[:TRAIN_DATA_NUM]
trainY = y[:TRAIN_DATA_NUM]

testX = X[TRAIN_DATA_NUM:]
testY = y[TRAIN_DATA_NUM:]

print("\nFinished loading and preparing the data\n"
      f"The number of sequences: {NUM_OF_SEQ}\n"
      f"The number of training sequences: {TRAIN_DATA_NUM}\n"
      f"The number of testing sequences: {TEST_DATA_NUM}\n")

print("#" * 72, "\n")

###############################################################################

# A placeholder for input, i.e. a matrix of shape SEQ_LENGTH x VOCAB_SIZE, the shape of the placeholder is (None, SEQ_LENGTH, VOCAB_SIZE), because None is reserved for batch size
input = tf.placeholder(tf.float32, shape=(
    None, SEQ_LENGTH, VOCAB_SIZE), name='input')

# A placeholder for targets, i.e. a matrix of shape SEQ_LENGTH x NUM_OF_CLASSES, the shape of the placeholder is (SEQ_LENGTH, NUM_OF_CLASSES)
targets = tf.placeholder(tf.float32, shape=(
    SEQ_LENGTH, NUM_OF_CLASSES), name='targets')

###############################################################################

# Create LSTM layers with forward and backward passes
# first_layer_outputs = create_lstm_layer_with_backward_pass(num_of_fw_lstms=LSTM_FW_NUMBER_FIRST_LAYER, num_of_bw_lstms=LSTM_BW_NUMBER_FIRST_LAYER, lstm_size=LSTM_SIZE,
#                                                            feedforward_units=FIRST_LAYER_UNITS, input=input, scope='lstm_first_layer', activation_function=tf.tanh, expand_dims=True)

# logits = create_lstm_layer_with_backward_pass(num_of_fw_lstms=LSTM_FW_NUMBER_SECOND_LAYER, num_of_bw_lstms=LSTM_BW_NUMBER_SECOND_LAYER, lstm_size=LSTM_SIZE,
#                                               feedforward_units=NUM_OF_CLASSES, input=first_layer_outputs, scope='lstm_second_layer', activation_function=None, expand_dims=False)


# Create LSTM layers with forward pass only
first_layer_outputs = create_lstm_layer(num_of_lstms=LSTM_ONLY_FW_NUMBER_FIRST_LAYER, lstm_size=LSTM_SIZE,
                                        feedforward_units=FIRST_LAYER_UNITS, input=input, scope='lstm_first_layer', activation_function=tf.tanh, expand_dims=True)

logits = create_lstm_layer(num_of_lstms=LSTM_ONLY_FW_NUMBER_SECOND_LAYER, lstm_size=LSTM_SIZE,
                           feedforward_units=NUM_OF_CLASSES, input=first_layer_outputs, scope='lstm_second_layer', activation_function=None, expand_dims=False)

###############################################################################

#  Pass the output from the network to the softmax activation function and compute the cross entropy after that
loss = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=targets, logits=logits)

# Create an operation for predicting after training - only softmax, no cross entropy, we do not want to calculate loss
prediction = tf.nn.softmax(logits, name='pred')

# Compute mean from loss to obtain one value to reflect the error of the network
cross_entropy = tf.reduce_mean(loss)

# Choose a method for performing backpropagation and tell it to minimize the loss
train_step = tf.train.RMSPropOptimizer(
    learning_rate=0.001, decay=0.9).minimize(cross_entropy)

# Output 1 if a character was correctly predicted and 0 if it was not
correct_prediction = tf.equal(tf.argmax(targets, 1),
                              tf.argmax(logits, 1))

# Compute the mean to get the accuracy
accuracy = (tf.reduce_mean(
    tf.cast(correct_prediction, tf.float32))) * 100

# Create a session for executing the graph
with tf.Session() as sess:
    # Initialize tf.Variables
    sess.run(tf.global_variables_initializer())

    for j in range(NUM_EPOCHS):
        # Shuffle the input data and the corresponding labels
        shuffle_indices = list(range(len(trainX)))
        shuffle(shuffle_indices)

        trainX_shuffled = [trainX[a] for a in shuffle_indices]
        trainY_shuffled = [trainY[a] for a in shuffle_indices]

        # Feed one SEQ_LENGTH at a time to the network. Since RNN demands batch size, as the first dimension, the list of inputs must be passed to it. Hence, trainX_shuffled[i] is enclosed in square brackets to make a one-item list from it. Feed_dict is an argument used for passing data to previously defined placeholders. Train_step is computed
        for i in range(TRAIN_DATA_NUM):
            sess.run(train_step,
                     feed_dict={input: [trainX_shuffled[i]], targets: trainY_shuffled[i]})

        # Compute the mean accuracy of the network after each epoch over the entire test dataset
        total_test_acc = 0

        for k in range(TEST_DATA_NUM):
            test_acc = sess.run(accuracy,
                                feed_dict={input: [testX[k]], targets: testY[k]})
            total_test_acc += test_acc

        print(
            f"Testing accuracy at epoch {j+1}: {(total_test_acc / TEST_DATA_NUM):.2f}")

    print('\n', "#" * 72, '\n')

    # After the training dump the decoded outputs to a file. The format is: <WORD> ---> <PREDICTED_LABEL> ---> <ACTUAL_LABEL>

    with open(OUTPUT_FILE, 'w') as g:

        padding = 25

        print("{0: <{3}}{1: <{3}}{2: <{3}}\n".format(
            "WORD", "PREDICTED_LABEL", "ACTUAL_LABEL", padding), file=g)

        for k in range(TEST_DATA_NUM):
            test_input = testX[k]
            test_labels = testY[k]

            out = sess.run(prediction, feed_dict={input: [test_input]})

            test_input_decoded = [
                ix_to_word[np.argmax(code)] for code in test_input]

            test_output_decoded = [
                ix_to_tag[np.argmax(fragment)] for fragment in out]

            test_labels_decoded = [
                ix_to_tag[np.argmax(label)] for label in test_labels]

            for l in range(SEQ_LENGTH):
                print(
                    f"{test_input_decoded[l]: <{padding}}{test_output_decoded[l]: <{padding}}{test_labels_decoded[l]: <{padding}}", file=g)
