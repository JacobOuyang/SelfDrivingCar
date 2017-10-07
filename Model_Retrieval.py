import tensorflow as tf
import numpy as np
import trainv2
import next_batch
class Inference():
    def __init__(self):
        self.sess = tf.Session()

    def restore_model(self):

        self.checkpoint_file = tf.train.latest_checkpoint('./runs/1499102036/checkpoints')
        self.new_saver = tf.train.import_meta_graph('{}.meta'.format(self.checkpoint_file))

        """all_vars = tf.global_variables()
        for v in all_vars:
            v_ = sess.run(v)
            print(v_)
        """
        graph = tf.get_default_graph()
        self.input_x = graph.get_operation_by_name('input_x').outputs[0]
        self.dropout_keep_prob = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
        self.input_y = graph.get_operation_by_name('input_y').outputs[0]
        self.scores = graph.get_operation_by_name('score').outputs[0]
        self.is_training = graph.get_operation_by_name("is_training").outputs[0]
        with self.sess.as_default():
            self.new_saver.restore(self.sess, self.checkpoint_file)


    def infer_batch_samples(self, x_test):


        with self.sess.as_default():
            batch_scores = self.sess.run(self.scores, {self.input_x: x_test, self.dropout_keep_prob: 1.0, self.is_training:False})
        return np.argmax(batch_scores, axis=1)

def main(_):
    model = Inference()
    model.restore_model()

    file_names = trainv2.load_file_names("/home/jacob/Desktop/Test10")
    batches = next_batch.next_batch(file_names, len(file_names), 1, 30)
    average = 0
    for batch in batches:
        images, labels = trainv2.load_images(batch)
        directions = model.infer_batch_samples(images)
        correct =0

        for i in range(len(labels)):
            print(directions[i] , '                ' , np.argmax(labels[i]))
            correct += np.argmax(labels[i] )==directions[i]
        print(correct)
        average += correct
    #true_direction = directions[0]
    print(average * 1.0 /len(file_names))
if __name__ == "__main__":
    tf.app.run(main=main)




