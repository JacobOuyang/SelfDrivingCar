import numpy
import math
def next_batch(file_names, num_examples, epochs, batch_size, shuffle=True):
    _epochs_completed = 0
    _index_in_epoch = 0
    number_of_batches = int(math.ceil(num_examples / batch_size))
    for _ in range(epochs):
        for batch_index in range(number_of_batches):

            start = _index_in_epoch
            if _epochs_completed == 0 and start == 0 and shuffle:
                perm0 = numpy.arange(num_examples)
                numpy.random.shuffle(perm0)
                shuffled_names = file_names[perm0]
            if start + batch_size > num_examples:
                _epochs_completed +=1
                rest_num_examples = num_examples - start
                batch_file_names = file_names[start:num_examples]

                if shuffle:
                    perm = numpy.arange(num_examples)
                    numpy.random.shuffle(perm)
                    shuffled_names = file_names[perm]

                start = 0
                _index_in_epoch = batch_size - rest_num_examples
                end = _index_in_epoch
                batch_file_names = shuffled_names[start:end]
            else:
                _index_in_epoch += batch_size
                end = _index_in_epoch
                batch_file_names= shuffled_names[start:end]
            yield batch_file_names



