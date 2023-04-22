from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D
from tensorflow.keras.layers import BatchNormalization, LSTM
import sys
import tensorflow as tf


def FullConn(
    units,
    activation=None,
    bias=None,
    use_bias=True,
    initializer="glo",
    name=None
):

    # weight initializer

    if initializer == "var":
        weight_initializer =  tf.keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
        )
    elif initializer == "glo":
        weight_initializer =  tf.keras.initializers.GlorotUniform()
    else:
        print(__file__, sys._getframe().f_lineno, "\n---------->", "Invalid initializer", initializer, "!!!")

    # bias initializer

    bias_initializer = tf.keras.initializers.Zeros()
    if bias is not None:
        bias_initializer = tf.keras.initializers.Constant(bias)
    
    return Dense(
        units,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=weight_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        name=name
    )
    

def Conv1Dim(
    input_shape, 
    filters, 
    kernel_size, 
    strides,
    bias=None,
    use_bias=True,
    padding="same",
    resnet=False,
    restype=None,
    initializer="glo",
    ignore_resnet=True
):

    # weight initializer
    if initializer == "var":
        weight_initializer =  tf.keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
        )
    elif initializer == "glo":
        weight_initializer =  tf.keras.initializers.GlorotUniform()
    else:
        print(__file__, sys._getframe().f_lineno, "\n---------->", "Invalid initializer", initializer, "!!!")
        raise Warning
    
    # bias initializer
    bias_initializer = tf.keras.initializers.Zeros()
    if bias is not None:
        bias_initializer = tf.keras.initializers.Constant(bias)

    if resnet and ignore_resnet == False:
        if restype == "block":
            # zeros initializer 
            weight_initializer = tf.keras.initializers.Zeros()
            pass
        elif restype == "skip":
            # ones initializer 
            weight_initializer = tf.keras.initializers.Zeros()
            bias_initializer = tf.keras.initializers.Constant(value=1)
            pass
        else:
            print(__file__, sys._getframe().f_lineno, "\n---------->", "Invalid initializer", initializer, "!!!")
            raise Warning
    
    if input_shape != None:
        return Conv1D(
            input_shape=input_shape,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None)
    else:
        return Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None)


def BatchNorm(
    momentum=0.99,
    epsilon=0.001,
    trainable=True
):
    return BatchNormalization(
        axis=-1,
        momentum=momentum,
        epsilon=epsilon,
        center=True,
        scale=True,
        beta_initializer='zeros',
        gamma_initializer='ones',
        moving_mean_initializer='zeros',
        moving_variance_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None,
    )


def LayerNorm(
    axis=-1,
    epsilon=0.001,
    center=True,
    scale=True,
    trainable=True
):
    return tf.keras.layers.LayerNormalization(
        axis=axis,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer='zeros',
        gamma_initializer='ones',
        beta_regularizer=None,
        gamma_regularizer=None,
        beta_constraint=None,
        gamma_constraint=None
    )

def MaxPool1Dim(
    pool_size,
    strides,
    padding="same",
):
    return MaxPool1D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format='channels_last'
    )


def RnnLSTM(
    units,
    bias=None,
    use_bias=True,
    activation='tanh',
    recurrent_activation='sigmoid',
    return_sequences=False, 
    return_state=False,
    initializer="glo",
    input_shape=None
):

    # weight initializer

    if initializer == "var":
        weight_initializer =  tf.keras.initializers.VarianceScaling(
            scale=1.0,
            mode="fan_in",
            distribution="normal",
        )
    elif initializer == "glo":
        weight_initializer =  tf.keras.initializers.GlorotUniform()
    else:
        print(__file__, sys._getframe().f_lineno, "\n---------->", "Invalid initializer", initializer, "!!!")

    # bias initializer

    bias_initializer = tf.keras.initializers.Zeros()
    if bias is not None:
        bias_initializer = tf.keras.initializers.Constant(bias)

    return LSTM(
        units=units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=weight_initializer,
        recurrent_initializer='orthogonal',
        bias_initializer=bias_initializer,
        unit_forget_bias=True,
        kernel_regularizer=None,
        recurrent_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        recurrent_constraint=None,
        bias_constraint=None,
        dropout=0.0,
        recurrent_dropout=0.0,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=False,
        stateful=False,
        time_major=False,
        unroll=False,
    )
