import math

def sigmoid(w_prod):
    sigmoid=1/(1+math.exp(-w_prod))
    return sigmoid

def error(exp_output,w_prod):
    error=exp_output-sigmoid(w_prod)
    return error

def update_weights(initial_weight,learning_rate,computed_error,input_x,w_prod):
    updated_weight= initial_weight+ learning_rate*computed_error*input_x*sigmoid(w_prod)*(1-sigmoid(w_prod))
    return updated_weight

