### Tensorflow implementation of  V-Net

This is a Tensorflow implementation of the ["V-Net"](https://arxiv.org/abs/1606.04797) architecture used for 3D medical 
imaging segmentation.


### How to use
The function `v_net(tf_input, input_channels, n_channels, output_channels)` receives a rank 5 tensor `tf_input` 
with shape `[batch_size, x, y, z, input_channels]` where `x, y, z` are the spatial dimensions of the images 
and `input_channels` is the number of channels the images have.

Apart from the number of input channels `input_channels` none of the dimensions of `tf_input` need to be known.

`n_channels` is the number of channels used internally in the network. In the original paper this number was 16.
This number doubles at every level of the contracting path. See the image for better understanding of this number.

`output_channels` allows you to specify the desired number of output channels. `v_net` will return a tensor with the 
same shape as `tf_input` but with a different number of channels *i.e.* `[batch_size, x, y, z, output_channels]`.
For example, in a binary segmentation problem you could use `output_channels=1` with a sigmoid loss and in a three class
problem you could use `output_channels=3` with a softmax loss.

### Implementation details

There are two different implementations.
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


### Use case

![VNetDiagram](VNetDiagram.png)






