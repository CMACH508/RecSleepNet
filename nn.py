import tensorflow as tf

def fc(
    name,
    inputs,
    n_hiddens,
    bias=None,
):
    # Weight initializer
    weight_initializer = tf.variance_scaling_initializer(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )
    # # MSRA initialization
    # weight_initializer = tf.contrib.layers.variance_scaling_initializer(
    #     factor=2.0,
    #     mode='FAN_IN',
    #     uniform=False
    # )

    # Determine whether to use bias
    use_bias = False
    bias_initializer = tf.zeros_initializer()
    if bias is not None:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias)

    # Dense
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.dense(
            inputs=inputs,
            units=n_hiddens,
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    return outputs
def conv1d(
    name,
    inputs,
    n_filters,
    filter_size,
    stride_size,
    bias=None,
    padding="SAME",
    activation = tf.nn.relu
):
    # Weight initializer
    weight_initializer = tf.variance_scaling_initializer(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )
    # # MSRA initialization
    # weight_initializer = tf.contrib.layers.variance_scaling_initializer(
    #     factor=2.0,
    #     mode='FAN_IN',
    #     uniform=False
    # )

    # Determine whether to use bias
    use_bias = False
    bias_initializer = tf.zeros_initializer()
    if bias is not None:
        use_bias = True
        bias_initializer = tf.constant_initializer(bias)

    # Convolution
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.conv2d(
            inputs=inputs,
            filters=n_filters,
            activation=activation,
            kernel_size=(filter_size,1),
            strides=(stride_size,1),
            padding=padding,
            data_format="channels_first",
            use_bias=use_bias,
            kernel_initializer=weight_initializer,
            bias_initializer=bias_initializer,
        )

    return outputs

#def sa(name, inputs, )
def max_pool1d(
    name,
    inputs,
    pool_size,
    stride_size,
    padding="SAME",
):
    # Max pooling
    if(inputs.shape[-2]%stride_size!=0):
        need  = stride_size - inputs.shape[-2]%stride_size
        inputs = tf.pad(inputs, [[0,0],[0,0],[0,need],[0,0]])
    #print(inputs.shape)
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.max_pooling2d(
            inputs,
            pool_size=(pool_size,1),
            strides=(stride_size,1),
            padding=padding,
            data_format="channels_first",
        )

    return outputs
def ave_pool1d(
    name,
    inputs,
    pool_size,
    stride_size,
    padding="SAME",
    data_format="channels_first",
):
    # Max pooling
    if(inputs.shape[-2]%stride_size!=0):
        need  = stride_size - inputs.shape[-2]%stride_size
        inputs = tf.pad(inputs, [[0,0],[0,0],[0,need],[0,0]])
    print(inputs.shape)
    with tf.variable_scope(name) as scope:
        outputs = tf.layers.average_pooling2d(
            inputs,
            pool_size=(pool_size,1),
            strides=(stride_size,1),
            padding=padding,
            data_format=data_format,
        )

    return outputs

def adam_optimizer_clip(
    loss,
    training_variables,
    global_step,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    clip_value=1.0,
):
    with tf.variable_scope("adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        grads_and_vars_op = optimizer.compute_gradients(
            loss=loss,
            var_list=training_variables
        )
        grads_op, vars_op = zip(*grads_and_vars_op)
        grads_op, _ = tf.clip_by_global_norm(grads_op, clip_value)
        apply_gradient_op = optimizer.apply_gradients(
            grads_and_vars=zip(grads_op, vars_op),
            global_step=global_step
        )
        return apply_gradient_op, grads_and_vars_op


def adam_optimizer(
    loss,
    training_variables,
    global_step,
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    with tf.variable_scope("adam_optimizer") as scope:
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                           beta1=beta1,
                                           beta2=beta2,
                                           epsilon=epsilon)
        grads_and_vars_op = optimizer.compute_gradients(
            loss=loss,
            var_list=training_variables
        )
        apply_gradient_op = optimizer.apply_gradients(
            grads_and_vars=grads_and_vars_op,
            global_step=global_step
        )
        return apply_gradient_op, grads_and_vars_op
