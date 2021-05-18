import tensorflow as tf
from ConcreteDropout import ConcreteDropout


# from tensorflow.keras.layers import Dense, Conv1D

def conv1d_with_concrete_dropout(x, out_dim, wd, dd):
    output = ConcreteDropout(tf.keras.layers.Conv1D(filters=out_dim,
                                                    kernel_size=1,
                                                    use_bias=True,
                                                    activation=None,
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True )(x, training=True)
    return output

def dense_with_concrete_dropout(x, out_dim, wd, dd):
    output = ConcreteDropout(tf.keras.layers.Dense(units=out_dim,
                                                   use_bias=True,
                                                   activation=None,
                                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                   bias_initializer=tf.contrib.layers.xavier_initializer()),
                             weight_regularizer=wd,
                             dropout_regularizer=dd,
                             trainable=True)(x, training=True)
    return output

def attn_matrix(A, X, attn_weight):
    # A : [batch, N, N]
    # X : [batch, N, F']
    # weight_attn : F'
    num_atoms = int(X.get_shape()[1])
    hidden_dim = int(X.get_shape()[2])
    _X1 = tf.einsum('ij,ajk->aik', attn_weight, tf.transpose(X, [0, 2, 1]))
    _X2 = tf.matmul(X, _X1)
    _A = tf.multiply(A, _X2)
    _A = tf.nn.tanh(_A)
    return _A


def get_gate_coeff(X1, X2, dim, label):
    num_atoms = int(X1.get_shape()[1])
    _b = tf.get_variable('mem_coef-' + str(label), initializer=tf.contrib.layers.xavier_initializer(), shape=[dim],
                         dtype=tf.float32)
    _b = tf.reshape(tf.tile(_b, [num_atoms]), [num_atoms, dim])

    X1 = tf.layers.dense(X1, units=dim, use_bias=False)
    X2 = tf.layers.dense(X2, units=dim, use_bias=False)

    output = tf.nn.sigmoid(X1 + X2 + _b)
    return output


def graph_attn_gate(A, X, attn, out_dim, label, length, num_train):
    X_total = []
    A_total = []
    wd = length ** 2 / num_train
    dd = 2. / num_train
    for i in range(len(attn)):
        _h = conv1d_with_concrete_dropout(X, out_dim, wd, dd)
        _A = attn_matrix(A, _h, attn[i])
        #print(_A.get_shape())

        _h = tf.nn.relu(tf.matmul(_A, _h))
        #print("hidden state shape",_h.get_shape())
        X_total.append(_h)
        A_total.append(_A)


    _X = tf.nn.relu(tf.concat(X_total, 2))
    _X = tf.layers.conv1d(_X,
                          filters=out_dim,
                          kernel_size=1,
                          use_bias=False,
                          activation=None,
                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                          bias_initializer=tf.contrib.layers.xavier_initializer())

    dim = int(_X.get_shape()[2])
    if (int(X.get_shape()[2]) != dim):
        X = tf.layers.dense(X, dim, use_bias=False)
    coeff = get_gate_coeff(_X, X, dim, label)
    output = tf.multiply(_X, coeff) + tf.multiply(X, 1.0 - coeff)
    return output


def encoder_gat_gate(X, A, num_layers, out_dim, num_attn, length, num_train):
    # X : Atomic Feature, A : Adjacency Matrix
    _X = X
    for i in range(num_layers):
        attn_weight = []
        for j in range(num_attn):
            attn_weight.append(tf.get_variable('eaw' + str(i) + '_' + str(j),
                                               initializer=tf.contrib.layers.xavier_initializer(),
                                               shape=[out_dim, out_dim],
                                               dtype=tf.float32)
                               )
            #print("attn weight",attn_weight)
        _X = graph_attn_gate(A, _X, attn_weight, out_dim, i, length, num_train)
        #print("avg",avg)
    return _X

#readout x is from the output of encoder gate, which is when all the state are updated
def readout_and_mlp(X, latent_dim, length, num_train):
    # X : [#Batch, #Atom, #Feature] --> Z : [#Batch, #Atom, #Latent] -- reduce_sum --> [#Batch, #Latent]
    # Graph Embedding in order to satisfy invariance under permutation
    wd = length ** 2 / num_train
    dd = 2. / num_train
    #print("x shape", X.get_shape())
    #Z = tf.nn.relu(conv1d_with_concrete_dropout(X, latent_dim, wd, dd))
    aggr = tf.random_normal([tf.shape(X)[0], 1,62])
    #print("aggr",aggr)

    Z=tf.keras.layers.Dense(1,activation=tf.keras.layers.LeakyReLU(0.1))(tf.concat([X,aggr],axis=1))
    #print("first dense",Z.get_shape())
    Z=tf.keras.layers.Flatten()(Z)
    #print("flatten:",Z.get_shape())
    Z=tf.nn.softmax(Z,axis=1)
    #print("softmax",Z)
    weights=Z
    Z=tf.keras.layers.RepeatVector(62)(Z)
    #print("Repeat vector:", Z)
    Z = tf.transpose(Z,[0,2,1])
    #print("transposed vector:", Z)
    Z=tf.keras.layers.multiply([(tf.concat([X,aggr],axis=1)),Z])
    #print("product:", Z)
    #i comment out this
    Z=tf.nn.sigmoid(tf.reduce_sum(Z,1))

    #print("z shape",Z.get_shape())
    #layer = (tf.keras.layers.Dense(1activation='sigmoid',kernel_initializer='he_uniform'))
    #outputs=layer(tf.transpose(Z,[0,2,1]))
    # output = tf.keras.layers.BatchNormalization()(outputs)
    # outputs = tf.keras.layers.AveragePooling1D(5,1,"same")(outputs)
    # outputs = tf.keras.layers.MaxPooling1D(11, 1, "same")(outputs)
    #print("outputs shape",outputs.get_shape)
    #Z=tf.squeeze(outputs,2)

    #print(layer.weights)
    #weights=layer.weights
    #Z=tf.concat([tf.nn.sigmoid(tf.reduce_sum(Z,1)), tf.nn.sigmoid(tf.reduce_mean(Z,1))],1)

    #print("z shape another:",Z.get_shape())
    # Predict the molecular property
    _Y = tf.nn.relu(dense_with_concrete_dropout(Z, latent_dim, wd, dd))
    #print("_y:", _Y.get_shape())
    #print("_y",_Y)
    Y_mean = tf.keras.layers.Dense(units=1,
                                   use_bias=True,
                                   activation=None,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    Y_logvar = tf.keras.layers.Dense(units=1,
                                     use_bias=True,
                                     activation=None,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     bias_initializer=tf.contrib.layers.xavier_initializer())(_Y)
    return Z, Y_mean, Y_logvar,weights
