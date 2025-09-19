import numpy as np


def softmax(vector):
    '''
    vector: np.array of shape (n, m)
    
    return: np.array of shape (n, m)
        Matrix where softmax is computed for every row independently
    '''
    nice_vector = vector - vector.max()
    exp_vector = np.exp(nice_vector)
    exp_denominator = np.sum(exp_vector, axis=1)[:, np.newaxis]
    softmax_ = exp_vector / exp_denominator
    return softmax_


def multiplicative_attention_score(decoder_hidden_state, W, encoder_hidden_state):
    intermediate = np.dot(decoder_hidden_state.T, W)
    attention_score = np.dot(intermediate, encoder_hidden_state)
    return attention_score


def multiplicative_attention(decoder_hidden_state, encoder_hidden_states, W_mult):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    W_mult: np.array of shape (n_features_dec, n_features_enc)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    scores = multiplicative_attention_score(decoder_hidden_state, W_mult, encoder_hidden_states)
    
    attention_weights = softmax(scores)
    
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector


def additive_attention_score(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    
    # Вычисляем: W_add_enc * h_i для каждого h_i
    W_enc_h = np.dot(W_add_enc, encoder_hidden_states) # (7, 5) × (5, 4) = (7, 4)
    
    # Вычисляем: W_add_dec * s
    W_dec_s = np.dot(W_add_dec, decoder_hidden_state) # (7, 3) × (3, 1) = (7, 1)
    
    # Складываем и применяем tanh: tanh(W_enc_h + W_dec_s)
    tanh_res = np.tanh(W_enc_h + W_dec_s) # (7, 4) + (7, 1) → broadcasting → (7, 4)
    
    # Вычисляем: v^T * tanh_result
    attention_score = np.dot(v_add.T, tanh_res) # (1, 7) × (7, 4) = (1, 4)
    
    return attention_score


def additive_attention(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec):
    '''
    decoder_hidden_state: np.array of shape (n_features_dec, 1)
    encoder_hidden_states: np.array of shape (n_features_enc, n_states)
    v_add: np.array of shape (n_features_int, 1)
    W_add_enc: np.array of shape (n_features_int, n_features_enc)
    W_add_dec: np.array of shape (n_features_int, n_features_dec)
    
    return: np.array of shape (n_features_enc, 1)
        Final attention vector
    '''
    scores = additive_attention_score(decoder_hidden_state, encoder_hidden_states, v_add, W_add_enc, W_add_dec)
    
    attention_weights = softmax(scores)
    attention_vector = np.dot(encoder_hidden_states, attention_weights.T)
    
    return attention_vector
