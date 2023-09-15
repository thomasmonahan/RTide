
def convnet(num_features):
    input_dims = (num_features - 1, 1)
    ins = num_features -1
    model = Sequential()
    model.add(Conv1D(filters=ins, kernel_size=1, activation=None,strides = 1, input_shape=input_dims, name='linear_layer'))
    model.add(Conv1D(filters=int(math.factorial(ins) / (math.factorial(2)*math.factorial(ins -2))), kernel_size=2, activation=None, strides=1, name='nonlinear_layer', kernel_regularizer=regularizers.l1(l=dropout)))
    model.add(Conv1D(filters=ins//2, kernel_size=3, activation='relu', name='nonlinear_layer2'))
    model.add(Flatten())
    model.add(Dense(1))
    return model


def nn(num_features):
    input_dims = num_features-1
    model_U = Sequential()
    model_U.add(Dense(input_dims, input_dim=input_dims, activation=None, name = 'linear_layer'))
    model_U.add(Dense(input_dims, activation='relu', name = 'nonlinear_layer'))
    model_U.add(keras.layers.Dropout(dropout))
    model_U.add(Dense(input_dims, activation='relu', name = 'nonlinear_layer2'))
    model_U.add(keras.layers.Dropout(dropout))
    model_U.add(Dense(1))
    return model

def load_model(model_type, num_features):
    if model_type == 'CNN':
        return convnet(num_features)
    elif model_type =='NN':
        return nn(num_features)
    else:
        raise ValueError('This type of model is not available in RTide yet')
