import numpy as np
import tensorflow as tf
import sys
import zipfile
import collections

#Word2Vec

def main():

	vocabulary = read_data('text8.zip')
	print('Data size', len(vocabulary))

	# Step 2: Build the dictionary and replace rare words with UNK token.
	vocab_size = 50000

	data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
	                                                        vocab_size)
	del vocabulary  # Hint to reduce memory.
	print('Most common words (+UNK)', count[:5])
	print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

	data_index = 0

'''
	batch_size = 20;
	#input data to be feeded in our feed dictionary
	training_data = tf.Placeholder(tf.int32, shape=[batch_size])
	labels_data = tf.Placeholder(tf.int32, shape= [batch_size, 1])

	#layers

	#layer1
	Units = 20;
	W1 = tf.Variable([vocab_size, Units]);
	b1 = tf.Variable(tf.zeros([vocab_size]))

	#embed = tf.MatMul(training_data, W1)
	embed = tf.embedding_lookup(vocab_size, units)

	#----layers end here -----

	#loss

	loss = tf.reduce.mean(tf.nn.nce_loss(weights = W1,
		biases = b1, labels = labels_data, input = embed, num_sampled = num_sampled, num_classes = vocab_size))

	optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

	session = tf.Session()

	for inputs, labels in generate_batch():
		feed_dict = {training_data:inputs , labels_data: labels}
		_, cur_loss = session.run([optimizer, loss], feed_dict = feed_dict)

'''

def read_data(filename):
	with zipfile.ZipFile(filename) as f:
		data = tf.compat.as_str(f.read(f.namelist()[0])).split()

	return data


def build_dataset(words, n_words):

	count = [['UNK', -1]]
	count.extend(collections.Counter(words).most_common(n_words - 1))
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
	  		index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reversed_dictionary

def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
    	target = skip_window  # target label at the center of the buffer
    	targets_to_avoid = [skip_window]
    	for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
		# Backtrack a little bit to avoid skipping words in the end of a batch
	data_index = (data_index + len(data) - span) % len(data)
	return batch, labels


if __name__ == "__main__":
	main()