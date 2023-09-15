## This can and will be condensed into one function but for the sake of sending
## the code out I left it together
def transfer_train_NN(df, mult_val, lati, train_days = 'all', lr = 0.000008):
    """
    Function for multistep training procedure using HA for first step.

    Parameters
    ----------
    df: DataFrame
        Contains data to train on
    mult_val: int/float
        number of timesteps per hours
    train_days: int (Optional)
        Used if training on only a subset of training data.
    lati: float
        Latitude of station for utide
    """
    ## mult_val == number of timesteps per hour
    ## mult_val can be inferred from response class
    ## train_days == number of days of data to train on.
    ## if train_days == 'all', training is done on all available data.
    ## Lati ==
    if train_days == 'all':
        length_val = len(dataset)
    else:
        length_val = train_days *24*mult_val

    #dataset = df.values[::-1]
    dataset = df.values
    num_features = np.shape(dataset)[1]

    shorttide = utide.solve(
    df[:length_val].index,
    df[:length_val].observations,
    lat=lati,
    method="ols",
    conf_int="MC",
    verbose=False,
    trend = False
    )

    short_pred = reconstruct(df[:].index, shorttide, verbose = False)
    utide_preds= short_pred['h']

    ### Setting up Linear Training
    train_X_LIN = dataset[:, 1:num_features]

    train_Y_LIN = utide_preds

    ### Setting up Global Training

    train_X = dataset[:length_val, 1:num_features]

    train_Y = dataset[:length_val, 0:1]


    test_X = dataset[length_val:, 1:num_features]

    test_Y = dataset[length_val:, 0:1]


    ## Loading Model
    model = load_model('NN', num_features)
    custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=stoppage,
    min_delta=0.0001,
    mode='min'
    )

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=15, min_lr=0.00000001)

    ## Only training Linear layer in first step
    model.get_layer('nonlinear_layer').trainable = False
    model.get_layer('nonlinear_layer2').trainable = False

    # Compile Model U
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mean_absolute_error', optimizer= optimizer)
    history = model.fit(train_X_LIN, train_Y_LIN, epochs=train_epochs, batch_size=b_size, verbose = 2, shuffle = True, validation_split = .3, callbacks=[custom_early_stopping,reduce_lr])
    if plot:
        if train_epochs > 0:
            plt.plot(history.history['loss'], label = 'Loss U')
            plt.plot(history.history['val_loss'], label = 'Val Loss U')
            plt.legend()
            plt.show()

    # Now lets retrain the nonlinear layer:
    model.get_layer('nonlinear_layer2').trainable = True
    model.get_layer('nonlinear_layer').trainable = True
    model.get_layer('linear_layer').trainable = False


    K.set_value(model.optimizer.learning_rate, lr)
    history2 = model.fit(train_X, train_Y, epochs=train_epochs2, batch_size=b_size, verbose = 2, shuffle = True, validation_split = .3, callbacks=[custom_early_stopping,reduce_lr])
    if plot:
        plt.plot(history2.history['loss'], label = 'Loss U')
        plt.plot(history2.history['val_loss'], label = 'Val Loss U')
        plt.legend()
        plt.show()

    ## Generating test predictions for different intervals: U
    test_predictions = model.predict(test_X)
    test_labels = test_Y

    train_predictions = model.predict(train_X)
    train_labels = train_Y

    return train_predictions, train_labels, test_predictions, test_labels


from keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers import Conv1D, Flatten, LSTM,MaxPooling1D,BatchNormalization
import math
def run_conv(df, train_days = 50, dim_reduc = 1, lr_1 = 0.000001, lr_2 = 0.000001, train_epochs = 10,train_epochs2 = 25, b_size = 64, plot = False, verbose = True, output = False, return_predictions = False, lati = 53.111032, mult_val = 4, dropout = 0, shift_val = 0, save = False, num_outs = 2, third = False, nonlin = False):
  length_val = train_days *24*mult_val

  dataset = df.values
  num_features = np.shape(dataset)[1]

  shorttide = utide.solve(
    df[:length_val].index,
    df[:length_val].observations,
    lat=lati,
    method="ols",
    conf_int="MC",
    verbose=False,
    trend = False
    )

  short_pred = reconstruct(df[:].index, shorttide, verbose = False)
  utide_preds= short_pred['h']
  utide_out_preds = reconstruct(df[length_val:].index, shorttide, verbose = False)['h']


  ### Setting up Linear Training
  train_X_LIN = dataset[: int(len(df)/dim_reduc), 1:num_features]
  train_X_LIN = train_X_LIN.reshape(train_X_LIN.shape[0], train_X_LIN.shape[1], 1)

  train_Y_LIN = utide_preds[:int(len(df)/dim_reduc)]

  ### Setting up Global Training

  train_X = dataset[:length_val, 1:num_features]
  train_X = train_X.reshape(train_X.shape[0], train_X.shape[1], 1)
  train_Y = dataset[:length_val, 0:1]


  test_X = dataset[length_val:, 1:num_features]
  test_X = test_X.reshape(test_X.shape[0], test_X.shape[1], 1)

  test_Y = dataset[length_val:, 0:1]



  scaler_X_LIN = StandardScaler()
  scaled_train_X_LIN = scaler_X_LIN.fit_transform(train_X_LIN.reshape(-1, 1)).reshape(train_X_LIN.shape)

  scaler_Y_LIN = StandardScaler()
  scaled_train_Y_LIN = scaler_Y_LIN.fit_transform(train_Y_LIN.reshape(-1, 1))


  scaler_X = StandardScaler()
  scaled_train_X = scaler_X.fit_transform(train_X.reshape(-1,1)).reshape(train_X.shape)
  scaled_test_X = scaler_X.transform(test_X.reshape(-1,1)).reshape(test_X.shape)
  scaler_Y = StandardScaler()
  scaled_train_Y = scaler_Y.fit_transform(train_Y.reshape(-1, 1))

  custom_early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    min_delta=0.0001,
    mode='min'
  )

  ## Setup Model
 model = load_model('CNN', num_features)

 reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=3, min_lr=0.00000001)




  optimizer = keras.optimizers.Adam(learning_rate=lr_1)
  model.compile(loss='mean_absolute_error', optimizer=optimizer)

  model.get_layer('nonlinear_layer').trainable = True
  model.get_layer('nonlinear_layer2').trainable = False

  # Compile Model U
  #optimizer = keras.optimizers.Adam(lr=lr_1)
  #model.compile(loss='mean_absolute_error', optimizer=optimizer)
  history = model.fit(scaled_train_X_LIN, scaled_train_Y_LIN, epochs=train_epochs, batch_size=b_size, verbose=2, shuffle=True, validation_split=.2, callbacks=[custom_early_stopping,reduce_lr])
  if train_epochs > 0:
    plt.plot(history.history['loss'], label = 'Loss U')
    plt.plot(history.history['val_loss'], label = 'Val Loss U')
    plt.legend()
    plt.show()

    # Now lets retrain the nonlinear layer:
  model.get_layer('nonlinear_layer2').trainable = True
  model.get_layer('nonlinear_layer').trainable = True
  model.get_layer('linear_layer').trainable = False


  history2 = model.fit(scaled_train_X, scaled_train_Y, epochs=train_epochs2, batch_size=b_size, verbose=2, shuffle=True, validation_split=.2, callbacks=[custom_early_stopping,reduce_lr])
  try:
    plt.plot(history2.history['loss'], label = 'Loss U')
    plt.plot(history2.history['val_loss'], label = 'Val Loss U')
    plt.legend()
    plt.show()
  except:
    pass

  train_predictions = scaler_Y.inverse_transform(model.predict(scaled_train_X))
  train_labels = df.values[:length_val, 0:1]

  test_predictions = model.predict(scaled_test_X)

  test_predictions = scaler_Y.inverse_transform(model.predict(scaled_test_X))
  test_labels = df.values[length_val:, 0:1]


  return model, train_predictions, train_labels, test_predictions, test_labels
