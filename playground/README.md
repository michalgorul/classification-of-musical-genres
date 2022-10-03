#### Stochastic gradient descent
Given a differentiable function, it’s theoretically possible to find its minimum analytically: it’s known that a function’s minimum is a point where the derivative is 0, so all you have to do is find all the points where the derivative goes to 0 and check for which of these points the function has the lowest value. Applied to a neural network, that means finding analytically the combination of weight values that yields the smallest possible loss function. This can be done by solving the equation gradient(f)(W) = 0 for W. This is a polynomial equation of N variables, where N is the number of coefficients in the network. Although it would be possible to solve such an equation for N = 2 or N = 3, doing so is intractable for real neural networks, where the number of parameters is never less than a few thousand and can often be several tens of millions. Instead, you can use the four-step algorithm outlined at the beginning of this section: modify the parameters little by little based on the current loss value on a random batch of data. Because you’re dealing with a differentiable function, you can compute its gradient, which gives you an efficient way to implement step 4. If you update the weights in the opposite direction from the gradient, the loss will be a little less every time:
1. Draw a batch of training samples x and corresponding targets y.
2. Run the network on x to obtain predictions y_pred.
3. Compute the loss of the network on the batch, a measure of the
mismatch between y_pred and y.
4. Compute the gradient of the loss with regard to the network’s parameters
(a backward pass).
5. Move the parameters a little in the opposite direction from the gradient—
for example W = step * gradient—thus reducing the loss on the
batch a bit.

Note that a variant of the mini-batch SGD algorithm would be to draw a single sample and target at each iteration, rather than drawing a batch of data. This would be true SGD (as opposed to mini-batch SGD). Alternatively, going to the opposite extreme, you could run every step on all data available, which is called batch SGD. Each update would then be more accurate, but far more expensive. The efficient compromise between these two extremes is to use mini-batches of reasonable size.
#### Key attributes of tensors
- **Number of axes (rank)** — For instance, a 3D tensor has three axes, and a
matrix has two axes. This is also called the tensor’s ndim in Python
libraries such as Numpy.
- **Shape** — This is a tuple of integers that describes how many dimensions
the tensor has along each axis. For instance, the previous matrix example
has shape (3, 5), and the 3D tensor example has shape (3, 3, 5).
A vector has a shape with a single element, such as (5,), whereas a
scalar has an empty shape, ().
- **Data type** (usually called dtype in Python libraries)—This is the type
of the data contained in the tensor; for instance, a tensor’s type could be
float32, uint8, float64, and so on. On rare occasions, you may
see a char tensor. Note that string tensors don’t exist in Numpy (or in
most other libraries), because tensors live in preallocated, contiguous
memory segments: and strings, being variable length, would preclude the
use of this implementation.

#### Real-world examples of data tensors
- **Vector** data— 2D tensors of shape (samples, features)
- **Timeseries data or sequence data** — 3D tensors of shape (samples,
timesteps, features)
- **Images** — 4D tensors of shape (samples, height, width,
channels) or (samples, channels, height, width)
- **Video**— 5D tensors of shape (samples, frames, height,
width, channels) or (samples, frames, channels,
height, width)

#### Building your network
Example:
- `Dense(16, activation='relu')`
  - The argument being passed to each Dense layer (16) is the number of hidden units of the layer. A hidden unit is a dimension in the representation space of the layer. You can intuitively understand the dimensionality of your representation space as “how much freedom you’re allowing the network to have when learning internal representations.” Having more hidden units (a higher-dimensional representation space) allows your network to learn more-complex representations, but it makes the network more computationally expensive and may lead to learning unwanted patterns(patterns that will improve performance on the training data but not on the test data).
  - Dense layer with a `relu` activation implements the following chain of tensor operations:
`output = relu(dot(W, input) + b)`
We have three tensor operations here: a dot product (dot)
between the input tensor and a tensor named W; an addition (+) between the resulting 2D tensor and a vector b; and, finally, a relu operation. relu(x) is max(x, 0).

#### A loss function 
How the network will be able to measure its
performance on the training data, and thus how it will be able to steer
itself in the right direction.
#### An optimizer
The mechanism through which the network will update
itself based on the data it sees and its loss function.
#### Metrics to monitor during training and testing
A metric is a function that is used to judge the performance of your model.
Metric functions are similar to loss functions, except that the results from evaluating a metric are not used when training the model. Note that you may use any loss function as a metric. Example metrics:
- Accuracy metrics - Calculates how often predictions equal labels.
- Probabilistic metrics - Computes the crossentropy metric between the labels and predictions. This is the crossentropy metric class to be used when there are only two label classes (0 and 1).
- Regression metrics - Computes the mean squared error between `y_true` and `y_pred`.
- Classification metrics based on True/False positives & negatives - **For now not that important**
- Image segmentation metrics - Computes the mean Intersection-Over-Union metric. General definition and computation:Intersection-Over-Union is a common evaluation metric for semantic image segmentation. For an individual class, the IoU metric is defined as follows: `iou = true_positives / (true_positives + false_positives + false_negatives)`

### Binary classification problem (two output classes)
- You usually need to do quite a bit of preprocessing on your raw data in order to be able to feed it—as tensors—into a neural network. Sequences of words can be encoded as binary vectors, but there are other encoding options, too. 
- Stacks of `Dense` layers with `relu` activations can solve a wide range of problems (including sentiment classification), and you’ll likely use them frequently. 
- In a binary classification problem (two output classes), your network should end with a `Dense` layer with one unit and a `sigmoid` activation: the output of your network should be a scalar between 0 and 1, encoding a probability. 
- With such a scalar sigmoid output on a binary classification problem, the loss function you should use is `binary_crossentropy`. 
- The `rmsprop` optimizer is generally a good enough choice, whatever your problem. That’s one less thing for you to worry about.
- As they get better on their training data, neural networks eventually start overfitting and end up obtaining increasingly worse results on data they’ve never seen before. Be sure to always monitor performance on data that is outside of the training set.