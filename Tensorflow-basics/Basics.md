
The central unit of data in TensorFlow is the tensor.
A tensor consists of a set of primitive values shaped into an array of any number of dimensions.

A computational graph is a series of TensorFlow operations arranged into a graph of nodes. Each node takes zero or more tensors as inputs and produces a tensor as an output.


```python
import tensorflow as tf
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)
```

    (<tf.Tensor 'Const_4:0' shape=() dtype=float32>, <tf.Tensor 'Const_5:0' shape=() dtype=float32>)


We need to create a session in order to print the values of a node


```python
sess = tf.Session()
print(sess.run([node1, node2]))
```

    [3.0, 4.0]


Combining Tensor nodes with operations. In this case addition.


```python
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))
```

    ('node3: ', <tf.Tensor 'Add_1:0' shape=() dtype=float32>)
    ('sess.run(node3): ', 7.0)


External inputs can be provided using placeholders. A placeholder is a promise to provide a value later.


```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
```


```python
#Assigning values to a and b
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))
```

    7.5
    [ 3.  7.]


Doing more calculations on the existing placeholders


```python
add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b:4.5}))
```

    22.5


#did not understand
To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value:


```python
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
```

Variables need to be initialized first using certain operations or else they remain uninitialized


```python
init = tf.global_variables_initializer()
sess.run(init)

```

We can assign multiple values to the placeholder variable and get multiple results


```python
print(sess.run(linear_model, {x:[1,2,3,4]}))
```

    [ 0.          0.30000001  0.60000002  0.90000004]


Here we have created the model but we need to test the accuracy of the data using y variables - known values. 
We use loss function, which is the amount of error in the model or how apart the current model is from the provided data.
Here we use sum of square of distance as the loss function in this case.

linear_model - y creates a vector where each element is the corresponding example's error delta. We call tf.square to square that error. Then, we sum all the squared errors to create a single scalar that abstracts the error of all examples using tf.reduce_sum:


```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

    23.66


We calculate the value of W and b and reassign those values


```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))
```

    0.0


We guessed the "perfect" values of W and b, but the whole point of machine learning is to find the correct model parameters automatically. 

 TensorFlow provides optimizers that slowly change each variable in order to minimize the loss function. The simplest optimizer is gradient descent. It modifies each variable according to the magnitude of the derivative of loss with respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone. Consequently, TensorFlow can automatically produce derivatives given only a description of the model using the function tf.gradients. For simplicity, optimizers typically do this for you. For example,


```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))

```

    [array([-0.9999969], dtype=float32), array([ 0.99999082], dtype=float32)]


These values are very near to the actual values of 1 and -1 


```python
#Using tf.contrib.learn

import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use `numpy_input_fn`. We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x = np.array([1., 2., 3., 4.])
y = np.array([0., -1., -2., -3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x}, y, batch_size=4,
                                              num_epochs=1000)

# We can invoke 1000 training steps by invoking the `fit` method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did. In a real example, we would want
# to use a separate validation and testing data set to avoid overfitting.
estimator.evaluate(input_fn=input_fn)
```

    WARNING:tensorflow:Using temporary folder as model directory: /var/folders/r1/p8m9jtr94jvc_s82n7_j22k00000gn/T/tmpY0H7TQ
    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_num_ps_replicas': 0, '_keep_checkpoint_max': 5, '_tf_random_seed': None, '_task_type': None, '_environment': 'local', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x12081a650>, '_tf_config': gpu_options {
      per_process_gpu_memory_fraction: 1
    }
    , '_task_id': 0, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_evaluation_master': '', '_keep_checkpoint_every_n_hours': 10000, '_master': ''}
    WARNING:tensorflow:Rank of input Tensor (1) should be the same as output_rank (2) for column. Will attempt to expand dims. It is highly recommended that you resize your input, as this behavior may change.
    WARNING:tensorflow:From /Users/sshroff/.local/lib/python2.7/site-packages/tensorflow/contrib/learn/python/learn/estimators/head.py:1362: scalar_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
    Instructions for updating:
    Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.
    INFO:tensorflow:Create CheckpointSaverHook.
    INFO:tensorflow:Saving checkpoints for 1 into /var/folders/r1/p8m9jtr94jvc_s82n7_j22k00000gn/T/tmpY0H7TQ/model.ckpt.
    INFO:tensorflow:loss = 6.5, step = 1
    INFO:tensorflow:global_step/sec: 1323.72
    INFO:tensorflow:loss = 0.119848, step = 101
    INFO:tensorflow:global_step/sec: 1387.15
    INFO:tensorflow:loss = 0.0270273, step = 201
    INFO:tensorflow:global_step/sec: 1363.11
    INFO:tensorflow:loss = 0.00495651, step = 301
    INFO:tensorflow:global_step/sec: 1396.51
    INFO:tensorflow:loss = 0.00279715, step = 401
    INFO:tensorflow:global_step/sec: 1506.28
    INFO:tensorflow:loss = 0.000883442, step = 501
    INFO:tensorflow:global_step/sec: 1537.06
    INFO:tensorflow:loss = 0.000217933, step = 601
    INFO:tensorflow:global_step/sec: 1539.2
    INFO:tensorflow:loss = 3.15612e-05, step = 701
    INFO:tensorflow:global_step/sec: 1675.38
    INFO:tensorflow:loss = 3.23154e-05, step = 801
    INFO:tensorflow:global_step/sec: 2037.37
    INFO:tensorflow:loss = 1.54925e-05, step = 901
    INFO:tensorflow:Saving checkpoints for 1000 into /var/folders/r1/p8m9jtr94jvc_s82n7_j22k00000gn/T/tmpY0H7TQ/model.ckpt.
    INFO:tensorflow:Loss for final step: 4.48671e-06.
    WARNING:tensorflow:Rank of input Tensor (1) should be the same as output_rank (2) for column. Will attempt to expand dims. It is highly recommended that you resize your input, as this behavior may change.
    WARNING:tensorflow:From /Users/sshroff/.local/lib/python2.7/site-packages/tensorflow/contrib/learn/python/learn/estimators/head.py:1362: scalar_summary (from tensorflow.python.ops.logging_ops) is deprecated and will be removed after 2016-11-30.
    Instructions for updating:
    Please switch to tf.summary.scalar. Note that tf.summary.scalar uses the node name instead of the tag. This means that TensorFlow will automatically de-duplicate summary names based on the scope they are created in. Also, passing a tensor or list of tags to a scalar summary op is no longer supported.
    INFO:tensorflow:Starting evaluation at 2017-03-24-18:36:09
    INFO:tensorflow:Finished evaluation at 2017-03-24-18:36:09
    INFO:tensorflow:Saving dict for global step 1000: global_step = 1000, loss = 2.67274e-06
    WARNING:tensorflow:Skipping summary for global_step, must be a float or np.float32.





    {'global_step': 1000, 'loss': 2.6727394e-06}




```python

```
