### TensorFlow: Static Graphs
PyTorch autograd looks a lot like TensorFlow: in both frameworks we define a computational graph, and use automatic differentiation to compute gradients. The biggest difference between the two is that **TensorFlow’s computational graphs are static and PyTorch uses dynamic computational graphs**.

In TensorFlow, we define the computational graph once and then execute the same graph over and over again, possibly feeding different input data to the graph. **In PyTorch, each forward pass defines a new computational graph**.

Static graphs are nice because you can optimize the graph up front; for example a framework might decide to fuse some graph operations for efficiency, or to come up with a strategy for distributing the graph across many GPUs or many machines. If you are reusing the same graph over and over, then this potentially costly up-front optimization can be amortized as the same graph is rerun over and over.

One aspect where static and dynamic graphs differ is control flow. For some models we may wish to perform different computation for each data point; for example a recurrent network might be unrolled for different numbers of time steps for each data point; this unrolling can be implemented as a loop. With a static graph the loop construct needs to be a part of the graph; for this reason TensorFlow provides operators such as tf.scan for embedding loops into the graph. With dynamic graphs the situation is simpler: since we build graphs on-the-fly for each example, we can use normal imperative flow control to perform computation that differs for each input.

### PyTorch: nn
Computational graphs and autograd are a very powerful paradigm for defining complex operators and automatically taking derivatives; however for large neural networks raw autograd can be a bit too low-level.

When building neural networks we frequently think of arranging the computation into layers, some of which have learnable parameters which will be optimized during learning.

In TensorFlow, packages like **Keras, TensorFlow-Slim, and TFLearn** provide higher-level abstractions over raw computational graphs that are useful for building neural networks.

In PyTorch, the **nn package serves this same purpose**. The nn package defines a set of Modules, which are roughly equivalent to neural network layers. A Module receives input Tensors and computes output Tensors, but may also hold internal state such as Tensors containing learnable parameters. The **nn** package also **defines a set of useful loss functions** that are commonly used when training neural networks.

### PyTorch: optim
Up to this point we have updated the weights of our models by manually mutating the Tensors holding learnable parameters (with *torch.no_grad()* or *.data* to avoid tracking history in autograd). This is not a huge burden for simple optimization algorithms like stochastic gradient descent, but in practice we often train neural networks using more sophisticated optimizers like AdaGrad, RMSProp, Adam, etc.

The *optim* package in PyTorch abstracts the idea of an optimization algorithm and provides implementations of commonly used optimization algorithms.

In this example we will use the *nn* package to define our model as before, but we will optimize the model using the Adam algorithm provided by the *optim* package

### PyTorch: Control Flow + Weight Sharing
As an example of dynamic graphs and weight sharing, we **implement a very strange model**: a fully-connected ReLU network that **on each forward pass chooses a random number between 1 and 4** and uses that many hidden layers, reusing the same weights multiple times to compute the innermost hidden layers.

For this model we can use normal Python flow control to implement the loop, and we can implement weight sharing among the innermost layers by simply reusing the same Module multiple times when defining the forward pass.

We can easily implement this model as a Module subclass: