import numpy as np
import tensorflow as tf
from textblob import TextBlob
from random import shuffle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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


def load_and_parse_data(filename, seq_length, train_test_dataset_split_factor):

    with open(filename, 'r') as f:
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
    vocabulary_size = len(unique_words)
    # Number of labels
    num_of_labels = len(ix_to_tag)
    # Number of sequences that can be created from the input data
    num_of_sequences = len(words) - seq_length + 1
    # How much of the data is to be used for training
    train_data_num = int(train_test_dataset_split_factor * num_of_sequences)
    test_data_num = num_of_sequences - train_data_num

    # Data preparation, one-hot encoded words and labels
    X = np.zeros((num_of_sequences, seq_length, vocabulary_size))
    y = np.zeros((num_of_sequences, seq_length, num_of_labels))

    # +1 to include the last sequence
    for i in range(num_of_sequences):
        X_sequence = words_pos_tags_indexed[i:seq_length + i]
        X_sequence_ix = [word_to_ix[word] for word, tag in X_sequence]
        input_sequence = np.zeros((seq_length, vocabulary_size))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

        target_sequence = np.zeros((seq_length, num_of_labels))
        for j in range(seq_length):
            target_sequence[j][X_sequence[j][1]] = 1.
        y[i] = target_sequence

    # Divide the dataset into training and testing parts
    trainX = X[:train_data_num]
    trainY = y[:train_data_num]

    testX = X[train_data_num:]
    testY = y[train_data_num:]

    return vocabulary_size, num_of_labels, ix_to_word, word_to_ix, ix_to_tag, trainX, trainY, testX, testY, train_data_num, test_data_num, words


def build_text_annotation_model(is_unidirectional, lstm_size, fw_lstm_fst, bw_lstm_fst, fw_lstm_snd, bw_lstm_snd, feedforward_units_fst, feedforward_units_snd, data):
    if is_unidirectional:
        # Create LSTM layers with forward pass only
        first_layer_outputs = create_lstm_layer(num_of_lstms=fw_lstm_fst, lstm_size=lstm_size,
                                                feedforward_units=feedforward_units_fst, input=data, scope='lstm_first_layer', activation_function=tf.tanh, expand_dims=True)

        logits = create_lstm_layer(num_of_lstms=fw_lstm_snd, lstm_size=lstm_size,
                                   feedforward_units=feedforward_units_snd, input=first_layer_outputs, scope='lstm_second_layer', activation_function=None, expand_dims=False)
    else:
        # Create LSTM layers with forward and backward passes
        first_layer_outputs = create_lstm_layer_with_backward_pass(num_of_fw_lstms=fw_lstm_fst, num_of_bw_lstms=bw_lstm_fst, lstm_size=lstm_size,
                                                                   feedforward_units=feedforward_units_fst, input=data, scope='lstm_first_layer', activation_function=tf.tanh, expand_dims=True)

        logits = create_lstm_layer_with_backward_pass(num_of_fw_lstms=fw_lstm_snd, num_of_bw_lstms=bw_lstm_snd, lstm_size=lstm_size,
                                                      feedforward_units=feedforward_units_snd, input=first_layer_outputs, scope='lstm_second_layer', activation_function=None, expand_dims=False)

    return logits


def train_text_annotation_model(logits, targets, trainX, trainY, testX, testY, train_data_num, test_data_num, epochs):
    # Pass the output from the network to the softmax activation function and compute the cross entropy after that
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=targets, logits=logits)

    # Compute mean from loss to obtain one value to reflect the error of the network
    cross_entropy = tf.reduce_mean(loss)

    # Choose a method for performing backpropagation and tell it to minimize the loss
    train_step = tf.train.RMSPropOptimizer(
        learning_rate=0.001, decay=0.9).minimize(cross_entropy)

    # Output 1 if a character was correctly predicted and 0 if it was not
    correct_prediction = tf.equal(tf.argmax(targets, 1), tf.argmax(logits, 1))

    # Compute the mean to get the accuracy
    accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100

    # Create a session for executing the graph
    sess = tf.Session()

    # Initialize tf.Variables
    sess.run(tf.global_variables_initializer())

    for j in range(epochs):
        # Shuffle the input data and the corresponding labels
        # shuffle_indices = list(range(len(trainX)))
        # shuffle(shuffle_indices)

        # trainX_shuffled = [trainX[a] for a in shuffle_indices]
        # trainY_shuffled = [trainY[a] for a in shuffle_indices]

        # Feed one SEQUENCE_LENGTH at a time to the network. Since RNN demands batch size, as the first dimension, the list of inputs must be passed to it. Hence, trainX_shuffled[i] is enclosed in square brackets to make a one-item list from it. Feed_dict is an argument used for passing data to previously defined placeholders. Train_step is computed
        for i in range(train_data_num):
            sess.run(train_step,
                     feed_dict={input_data: [trainX[i]], targets: trainY[i]})

        # Compute the mean accuracy of the network after each epoch over the entire test dataset
        total_test_acc = 0

        for k in range(test_data_num):
            test_acc = sess.run(accuracy,
                                feed_dict={input_data: [testX[k]], targets: testY[k]})
            total_test_acc += test_acc

        print(
            f"Testing accuracy at epoch {j+1}: {(total_test_acc / test_data_num):.2f}")

    return sess


def predict_text_annotation_model(logits, ix_to_word, word_to_ix, seq_length, num_of_labels, vocabulary_size, original_dataset, input_file, output_file, sess):
    # Create an operation for predicting after training - only softmax, no cross entropy, we do not want to calculate loss
    prediction = tf.nn.softmax(logits, name='pred')

    print("\nStarted loading and parsing the data...")

    with open(input_file, 'r') as f:
        data = f.read()

    # Create a TextBlob object from the input data
    data = TextBlob(data)
    words = data.words

    # Remove words that were not present in the training and test dataset
    words = [word for word in words if word in original_dataset]

    # Number of sequences that can be created from the input data
    num_of_sequences = len(words) - seq_length + 1

    # Data preparation, one-hot encoded words and labels
    X = np.zeros((num_of_sequences, seq_length, vocabulary_size))
    for i in range(num_of_sequences):
        X_sequence = words[i:seq_length + i]
        X_sequence_ix = [word_to_ix[word] for word in X_sequence]
        input_sequence = np.zeros((seq_length, vocabulary_size))
        for j in range(seq_length):
            input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence

    print("\nFinished loading and parsing the data!")

    print(
        f"\nPrinting words and their predicted labels to the '{output_file}' file...")

   # After the training dump the decoded outputs to a file. The format is: <WORD> <PREDICTED_LABEL> <ACTUAL_LABEL>

    with open(output_file, 'w') as g:

        padding = 25

        print("{0: <{2}}{1: <{2}}\n".format(
            "WORD", "PREDICTED_LABEL", padding), file=g)

        for k in range(num_of_sequences):
            prediction_input = X[k]

            out = sess.run(prediction, feed_dict={
                           input_data: [prediction_input]})

            prediction_input_decoded = [
                ix_to_word[np.argmax(code)] for code in prediction_input]

            prediction_output_decoded = [
                ix_to_tag[np.argmax(fragment)] for fragment in out]

            # Print only the first element of every sequence to avoid duplications
            print(
                f"{prediction_input_decoded[0]: <{padding}}{prediction_output_decoded[0]: <{padding}}", file=g)


# Size of tensor containing hidden state of the cell, does not have any correlation with input, output or target
LSTM_SIZE = 128
# Number of LSTM cells to be used in the first layer
LSTM_FW_NUMBER_FIRST_LAYER = 2
LSTM_BW_NUMBER_FIRST_LAYER = 2
LSTM_ONLY_FW_NUMBER_FIRST_LAYER = 4

# Number of LSTM cells to be used in the second layer
LSTM_FW_NUMBER_SECOND_LAYER = 1
LSTM_BW_NUMBER_SECOND_LAYER = 1
LSTM_ONLY_FW_NUMBER_SECOND_LAYER = 2

# Number of output units from the feedforward layer in the first layer
FIRST_LAYER_UNITS = 64

print("""
    Welcome to the Part-of-Speech Tagger! Remember to follow the right sequence of instructions (load -> train -> predict). Type the following commands to get started:

    load    ---> load the training and testing data and build the model
    train   ---> train the model
    predict ---> generate predictions and write them to a file
    quit    ---> quit the program
    """)

while True:
    ans = input("What would you like to do? ---> ")
    print()
    try:

        if ans == "load":

            filename = input("Type the name of the file with text inside: ")
            seq_len = int(input("Enter the length of a sequence: "))
            train_test_set_split = float(input(
                "Enter a number specifying what part of the dataset should be used for training: "))
            is_unidirectional = input(
                "Should the model be unidirectional? [y/n]: ")

            if (is_unidirectional == 'y'):
                is_unidirectional = True
            else:
                is_unidirectional = False

            print("\nStarting loading and preparing the data...")

            vocabulary_size, num_of_labels, ix_to_word, word_to_ix, ix_to_tag, trainX, trainY, testX, testY, train_data_num, test_data_num, original_dataset = load_and_parse_data(
                filename, seq_len, train_test_set_split)

            print("\nFinished loading and preparing the data!")

            print("\nStarting building the model...")

            tf.reset_default_graph()

            # A placeholder for input, i.e. a matrix of shape seq_len x vocabulary_size
            # , the shape of the placeholder is (None, seq_len, vocabulary_size
            # ), because None is reserved for batch size
            input_data = tf.placeholder(tf.float32, shape=(
                None, seq_len, vocabulary_size
            ), name='input')

            # A placeholder for targets, i.e. a matrix of shape seq_len x num_of_labels, the shape of the placeholder is (seq_len, NUM_OF_LABELS)
            targets = tf.placeholder(tf.float32, shape=(
                seq_len, num_of_labels), name='targets')

            logits = build_text_annotation_model(is_unidirectional, LSTM_SIZE, LSTM_FW_NUMBER_FIRST_LAYER, LSTM_BW_NUMBER_FIRST_LAYER,
                                                 LSTM_FW_NUMBER_SECOND_LAYER, LSTM_BW_NUMBER_SECOND_LAYER, FIRST_LAYER_UNITS, num_of_labels, input_data)

            print("\nFinished building the model!")

        elif ans == "train":

            num_epochs = int(input(
                "Specify how many epochs should the training last: "))

            print("\nStarting training phase...\n")

            sess = train_text_annotation_model(
                logits, targets, trainX, trainY, testX, testY, train_data_num, test_data_num, num_epochs)

            print("\nFinished training phase!\n")

        elif ans == "predict":

            input_file = input(
                "Type the name of the file with input data: ")

            output_file = input(
                "Type the name of the file to write predictions to: ")

            predict_text_annotation_model(
                logits, ix_to_word, word_to_ix, seq_len, num_of_labels, vocabulary_size, original_dataset, input_file, output_file, sess)

            print("\nFinished printing the words and the labels!\n")

        elif ans == "quit":

            break

    except:
        print(
            "\nSorry, something went wrong :( Please, try again!\n")

    print("\n", "#" * 72, "\n\n")

print("\nThank you for using me! Have a jolly good day, goodbye :)\n")
