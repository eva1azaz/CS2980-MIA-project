'''
Created on 15 Nov 2017

@author: ahmed.salem
Based on https://github.com/csong27/membership-inference/blob/master/attack.py
'''

import sys

sys.dont_write_bytecode = True

################## COMMENT OUT WHICHEVER MODEL IS NOT IN USE ##################
########### classifier: base model; dropout_classifier: dropout model ##########
#################################################################################

from classifier import train_model, iterate_minibatches
# from dropout_classifier import train_model, iterate_minibatches

#################################################################################
#################################################################################

import numpy as np
import theano.tensor as T
import lasagne
import theano



np.random.seed(21312)



def train_target_model(dataset,epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', maskConfidences=False):

    
    train_x, train_y, test_x, test_y = dataset

    output_layer = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio)
    # test data for attack model
    attack_x, attack_y = [], []
    if model=='cnn':
        #Dimension for CIFAR-10
        input_var = T.tensor4('x')
    else:
        #Dimension for News
        input_var = T.matrix('x')

    prob = lasagne.layers.get_output(output_layer, input_var, deterministic=True)
    prob_fn = theano.function([input_var], prob) # prob_fn is a function that will output the confidence scores (i.e. softmax output probabilities)
    
    def mask_confidences(probs, clip_val=0.9):
        probs = np.clip(probs, 0, clip_val)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs

    # data used in training, label is 1
    for batch in iterate_minibatches(train_x, train_y, batch_size, False):

        # CONFIDENCE MASKING: mask the confidence scores to make them more similar to the test data
        # attack_x.append(prob_fn(batch[0]))
        raw_probs = prob_fn(batch[0])
        if maskConfidences:
            raw_probs = mask_confidences(raw_probs)
            print("Masked max confidence:", np.max(raw_probs[0]))  # should be <= 0.9
        attack_x.append(raw_probs)
        attack_y.append(np.ones(len(batch[0])))
        
    # data not used in training, label is 0
    for batch in iterate_minibatches(test_x, test_y, batch_size, False):

        # store the posterior probability vectors (confidence scores) into the dataset 
        # this is the data used in training the attack model, label is 0
        attack_x.append(prob_fn(batch[0])) # batch[0] are the input samples
        attack_y.append(np.zeros(len(batch[0])))
        
    #print len(attack_y)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    #print('total length  ' + str(sum(attack_y)))
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    return attack_x, attack_y, output_layer


