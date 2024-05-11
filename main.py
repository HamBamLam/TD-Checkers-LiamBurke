import os
import tensorflow as tf

from model import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

#flags for picking whether to test agent or use "play" function, and one on whether to restore previous agent 
flags.DEFINE_boolean('test', False, 'If true, test against a random strategy.')
flags.DEFINE_boolean('draw', False, 'If true, play against a trained TD-Gammon strategy.')
flags.DEFINE_boolean('restore', False, 'If true, restore a checkpoint before training.')

#path for holding model, summary, checkpoints
model_path = os.environ.get('MODEL_PATH', 'models/')
summary_path = os.environ.get('SUMMARY_PATH', 'summaries/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
#tracking path was for logs file, decided to just use python filewriter instead
#tracking_path = os.environ.get('TRACKING_PATH', 'tracking/')

if not os.path.exists(model_path):
    os.makedirs(model_path)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)

if not os.path.exists(summary_path):
    os.makedirs(summary_path)
    
#if not os.path.exists(tracking_path):
    #os.makedirs(tracking_path)

if __name__ == '__main__':
    #build graph, build session, then create model
    graph = tf.Graph()
    sess = tf.Session(graph=graph)
    with sess.as_default(), graph.as_default():
        model = Model(sess, model_path, summary_path, checkpoint_path, restore=FLAGS.restore)
        if FLAGS.test:
            model.test(episodes=100)
        elif FLAGS.draw:
            model.draw()
        else:
            model.train()
