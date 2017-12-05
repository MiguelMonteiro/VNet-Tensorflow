### Tensorflow implementation of  V-Net

This is a Tensorflow implementation of the ["V-Net"](https://arxiv.org/abs/1606.04797) architecture used for 3D medical 
imaging segmentation.
This code only implements the Tensorflow graph, it must be used within a training program.

### Visual Representation of the Network

This is an example a network this code implements.

![VNetDiagram](VNetDiagram.png)

### How to use

The function `v_net(tf_input, input_channels, output_channels, n_channels)` has the following arguments:

1. `tf_input`: a rank 5 tensor with shape `[batch_size, X, Y, Z, input_channels]` where `X, Y, Z` are the spatial 
dimensions of the images and `input_channels` is the number of channels the images have;

2. `input_channels`: the number of channels of the input images;

3. `output_channels` is the number of desired output channels. `v_net()` will return a tensor with the 
same shape as `tf_input` but with a different number of channels *i.e.* `[batch_size, x, y, z, output_channels]`.

4. `n_channels` is the number of channels used internally in the network. In the original paper this number was 16.
This number doubles at every level of the contracting path. See the image for better understanding of this number.

##### Notes

Apart from the number of input channels `input_channels` none of the dimensions of `tf_input` need to be known.
This allows reading examples from queues and even train the network with examples of different sizes.

In a binary segmentation problem you could use `output_channels=1` with a sigmoid loss and in a three class
problem you could use `output_channels=3` with a softmax loss.

### Example Usage

````
import tensorflow as tf
from VNet import v_net

input_channels = 6
ouptut_channels = 1
 
tf_input = tf.placeholder(dtype=tf.float32, shape=(10, 190, 190, 20, input_channels))

logits = v_net(tf_input, input_channels, 16, output_channels)

````
`logits` will have shape `[10, 190, 190, 20, 1]`, it can the be flattened and used in the sigmoid cross entropy function.


### Implementation details

There are two different slightly implementations in the code.

`VNetOriginal.py` implements the network as is in the [original paper]((https://arxiv.org/abs/1606.04797)) with 
three small differences.
1. The input can have more than one channel. If the input has more than one channel han one more convolution is added 
in level one to increase the input number of channels to match `n_channels`. If the input has only one channel than it 
is broadcasted in the first skip connection (repeated ``n_channel` times).
2. `n_channels` does not need to be 16.
3. The output does not need to have two channels like in the original architecture.

If you `input_channels=1`, `n_channels=16` and `output_channels=2` the function `v_net()` implements the 
original architecture.

`VNet.py` is an updated version of the architecture with the following improvements/fixes:
 1. Relu non-linearities replaced with PRelu (Parametric Relu)
 2. The ["V-Net" paper](https://arxiv.org/abs/1606.04797) implemented the element-wise sum of the skip connection after the non-linearity.
 However, according to the [original residual network paper](https://arxiv.org/abs/1512.03385), 
 this should be done before the last non-linearity of the convolution block.
 This is fixed in this implementation.







