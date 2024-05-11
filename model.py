from __future__ import division

import time
import random
import copy
import numpy as np
import tensorflow as tf
from checkers.game_liam import Game
import tensorflow as tf
from checkers.agents.td_agent import TDAgent
from checkers.agents.random_agent import RandomAgent


'''
TensorFlow logic is adapted heavily from TDGammon, since my experience with Tensorflow is not great. 
I realized pretty quickly that I was out of my depth with trying to implement a checkers version of TDGammon
on my own, since I wasn't even aware of packages like TensorFlow and Pytorch and was trying to set everything up with default Python.
Thus I decided to use the TDGammon implementation and adapt it to Checkers instead, with necessary changes like how the game is
represented and how it flows, as well as implementing an alpha-beta pruning search, simple features, etc.

Basic flow is features as input, feed features into sigmoid layers, get V_next as output from layers.
get gradients and update weights based on them
eligibility trace is implemented with gradient, gradient updates then applied in one step

Training is done by playing against self, where a large number of episodes are run with a random starting player.
On each step, features and output are determined and the session is run to apply updates. After each episode,
the session is run to show error/loss statistics/summaries. On every validation interval, the agent is tested
against a random agent to see how the new model performs. After a training session, the new agent is stored
in a checkpoints folder to be loaded later

'''

# helper to initialize a weight and bias variable
def weight_bias(shape):
    W = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=shape[-1:]), name='bias')
    return W, b

# helper to create a dense, fully-connected layer
def dense_layer(x, shape, activation, name):
    with tf.variable_scope(name):
        W, b = weight_bias(shape)
        return activation(tf.matmul(x, W) + b, name='activation')

class Model(object):
    def __init__(self, sess, model_path, summary_path, checkpoint_path, tracking_path=None, restore=False):
        #required paths
        self.model_path = model_path
        self.summary_path = summary_path
        self.checkpoint_path = checkpoint_path
        #self.tracking_path = tracking_path
        
        self.re = restore

        # setup our session
        self.sess = sess
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # lambda decay, from 0.9 to 0.7, starting at step 30000, with a rate of 0.96
        lamda = tf.maximum(0.7, tf.train.exponential_decay(0.9, self.global_step, \
            30000, 0.96, staircase=True), name='lambda')

        # learning rate decay, from .1 to .01, meant to produce more stable learning
        alpha = tf.maximum(0.01, tf.train.exponential_decay(0.1, self.global_step, \
            30000, 0.96, staircase=True), name='alpha')

        tf.summary.scalar('lambda', lamda)
        tf.summary.scalar('alpha', alpha)

        # describe network size
        layer_size_input = 69 #length of feature vector
        layer_size_hidden = 20 #moderate size to avoid having too little or too many
        layer_size_output = 1 #output meant to be evaluation, length of 1
        # input and output
        self.x = tf.placeholder('float', [1, layer_size_input], name='x')
        self.V_next = tf.placeholder('float', [1, layer_size_output], name='V_next')
        # build network arch. (2 sigmoid layers)
        prev_y = dense_layer(self.x, [layer_size_input, layer_size_hidden], tf.sigmoid, name='layer1')
        self.V = dense_layer(prev_y, [layer_size_hidden, layer_size_output], tf.sigmoid, name='layer2')
        # watch the V and V_next over episodes
        tf.summary.scalar('V_next', tf.reduce_sum(self.V_next))
        tf.summary.scalar('V', tf.reduce_sum(self.V))
        # delta = V_next - V
        delta_op = tf.reduce_sum(self.V_next - self.V, name='delta')
        #track game step
        with tf.variable_scope('game'):
            game_step = tf.Variable(tf.constant(0.0), name='game_step', trainable=False)
            game_step_op = game_step.assign_add(1.0)
            # reset per-game monitoring variables
            game_step_reset_op = game_step.assign(0.0)
            self.reset_op = tf.group(*[game_step_reset_op])

        # increment global step: we keep this as a variable so it's saved with checkpoints
        global_step_op = self.global_step.assign_add(1)

        # get gradients of output V and trainable variables (weights and biases)
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.V, tvars)
        # for each variable, define operations to update the var with delta,
        # taking into account the gradient as part of the eligibility trace
        apply_gradients = []
        with tf.variable_scope('apply_gradients'):
            for grad, var in zip(grads, tvars):
                with tf.variable_scope('trace'):
                    # e-> = lambda * e-> + <grad of output w.r.t weights>
                    trace = tf.Variable(tf.zeros(grad.get_shape()), trainable=False, name='trace')
                    trace_op = trace.assign((lamda * trace) + grad)
                    tf.summary.histogram(var.name + '/traces', trace)

                # grad with trace = alpha * delta * e
                grad_trace = alpha * delta_op * trace_op
                tf.summary.histogram(var.name + '/gradients/trace', grad_trace)

                grad_apply = var.assign_add(grad_trace)
                apply_gradients.append(grad_apply)

        # as part of training we want to update our step
        with tf.control_dependencies([
            global_step_op,
            game_step_op,
        ]):
            # define single operation to apply all gradient updates
            self.train_op = tf.group(*apply_gradients, name='train')

        # merge summaries for TensorBoard
        self.summaries_op = tf.summary.merge_all()

        # create a saver for periodic checkpoints
        self.saver = tf.train.Saver(max_to_keep=1)

        # run variable initializers
        self.sess.run(tf.global_variables_initializer())

        # after training a model, we can restore checkpoints here
        if restore:
            self.restore()
    #restore latest checkpoint
    def restore(self):
        latest_checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_path)
        if latest_checkpoint_path:
            print('Restoring checkpoint: {0}'.format(latest_checkpoint_path))
            self.saver.restore(self.sess, latest_checkpoint_path)

    #get output from features 'x'
    def get_output(self, x):
        return self.sess.run(self.V, feed_dict={ self.x: x })
    
    #currently just used to see how agent actually plays
    def draw(self):
        game = Game.new()
        game.play([TDAgent(Game.PAWNS[0], self), RandomAgent(Game.PAWNS[1])], draw=True)

    def test(self, episodes=100):
        players = [TDAgent(Game.PAWNS[0], self), RandomAgent(Game.PAWNS[1])]
        winners = [0, 0]
        for episode in range(episodes):
            game = Game.new()

            winner = game.play(players)
            if winner != 2:
                winners[winner] += 1

            winners_total = sum(winners)
            print("[Game %d] %s vs %s, record of %d:%d" % (episode,players[0].name,players[1].name,winners[0], winners[1]))
        return (winners[0] / winners_total) * 100.0

    def train(self):
        tf.train.write_graph(self.sess.graph_def, self.model_path, 'td_gammon.pb', as_text=False)
        summary_writer = tf.summary.FileWriter('{0}{1}'.format(self.summary_path, int(time.time()), self.sess.graph_def))
        players = []
        episodes = 5000
        validation_interval = 500
        #copy_interval = 500
        log_file_path = 'logs.txt'
        logs = open(log_file_path, 'a')
        logs.write("-----------------------NEW TRAINING SESSION--------------\n")
        logs.flush()
        # the agent plays against random
        if not self.re:
            players = [TDAgent(Game.PAWNS[0], self), RandomAgent(Game.PAWNS[1])]
            episodes = 10000
            validation_interval = 1000
        else:
            #pit agent against self, meant to train to beat self            
            players = [TDAgent(Game.PAWNS[0], self), TDAgent(Game.PAWNS[1], self)]

        for episode in range(episodes):
            if episode != 0 and episode % validation_interval == 0: #validation interval lets us test agent while training
                win_percentage = self.test(episodes=100)
                logs.write("Episode: {}, Win Percentage: {}\n".format(episode, win_percentage))
                logs.flush()
            if episode == 5000 and not self.re:
                players = [TDAgent(Game.PAWNS[0], self), TDAgent(Game.PAWNS[1], self)]
                logs.write("-----------------------SWITCH TO SELF TRAINING--------------\n")
                logs.flush()
            game = Game.new()
            player_num = random.randint(0, 1)
            x = game.extract_features(players[player_num].player)
            max_turns = 500 #prevent end game loops from causing infinite loop
            game_step = 0
            #at each step, switch player num, get features and output of step, then train based on prev step
            while (not game.is_over(player_num)) and game_step < max_turns:
                game.next_step(players[player_num], player_num)
                player_num = (player_num + 1) % 2
                x_next = game.extract_features(players[player_num].player)
                #get approximation from model
                V_next = self.get_output(x_next)
                #run sess to update model
                self.sess.run(self.train_op, feed_dict={ self.x: x, self.V_next: V_next })
                x = x_next
                game_step += 1
            winner = (player_num+1) % 2
            win_string = players[winner].player
            if game_step >= max_turns:
                win_string = "Draw" 
            #run sess to update model after game end 
            _, global_step, summaries, _ = self.sess.run([
                self.train_op,
                self.global_step,
                self.summaries_op,
                self.reset_op,
            ], feed_dict={ self.x: x, self.V_next: np.array([[-1*winner+1]], dtype='float') })
            summary_writer.add_summary(summaries, global_step=global_step)
            print("Game %d/%d (Winner: %s) in %d turns" % (episode, episodes, win_string, game_step))
            self.saver.save(self.sess, self.checkpoint_path + 'checkpoint', global_step=global_step)
        summary_writer.close()
        win_percentage = self.test(episodes=1000)
        logs.write("Final Test, Win Percentage: {}\n".format(win_percentage))
        logs.flush()

