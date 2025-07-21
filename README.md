

[CIFAR DLRL Summer School](https://dlrl.ca/)

### Neural Networks

Sarath Chandar: Canada CIFAR AI Chair; Canada Research Chair in Lifelong ML
sarath.chandar@mila.quebec
https://chandar-lab.github.io

i.i.d. assumption: the data is identically distributed (which data?)

Logistic regression: why is it not called classification?

Maximising likelihood: $\max p(t | x)$. And so on... until negative log likelihood.
TODO Insert this sequence

Does minibatch usage help to alleviate the problem with local minimum?
Why? Gradients are noisy, and this noise helps to avoid local minimums.
Is there a formal proof?

TODO Insert he sequence to get the differentiation of obj. loss.

What can we model with logistic regression? Linearly separable; linear decision boundary.
TODO Insert the proof from the slides.
In literature: generalised linear models.

Neural Network aka MLP aka feed-forward NN aka fully connected NN

It is important that $g(.)$ is a non-linear activation function (otherwise this becomes the linear model).

Universal approximation theorem (Harnik 1991): "given enough hidden units".
This is an existential theorem.

Usually there is a gap between what NN can represent and what the optimisation techniques can learn.

Deep NNs: instead of having NN with exponential hidden units, create more layers.

Training deep NN: in 90s people tried 5 layers or more, but NNs didn't simply learn anything. NN winter. SVMs became popular.
Vanishing gradient problem: $\sigma'$ is at max 0.25.

How to train deep NNs? Tricks:
1. Greedy layer-wise pre-training (Hinton 2006, Bengio 2006). Learn hidden layer by auto-encoding: x -> h_1 -> x; then throw away the decoder and learn another auto-encoder: x -> h1 -> h2 -> h1 where w(1) for weights before h1 are frozen; and so on.
2. Pre-training: ResNet, word embeddings, BERT, ViT, GPT, llama.
3. Different activation functions: sigmoid, tanh -> ReLU, Leaky ReLU, GLU (Gated Linear Unit): what is the formula?, SwiGLU (what is it?)
4. Normalization: Batch Norm (Ioffe & Szegedy, 2014). Problem with ReLU: exploding gradients; problem with tanh: vanishing gradients, bad and good regions. Layer Norm. RMS Norm (what is it?). By default: RMS Norm.
5. Skip connections: Highway Networks, Residual Networks, Modern GLUs.
6. Adaptive gradients: if gradients too large -> decrease alpha & vice-versa. RMSProp, Adam. (Advice: don't use just Adam, try different optimizers.)

Why DL works?
Usually models overfit with increase in model complexity. But not with DL. Why? Model complexity was usually defined as the number of parameters.
Expressivity vs learnability. You can't just count the number of params in NN. Overparameterization works. There is a lot of theory (which one?)
Lottery ticket hypothesis. TODO: understand this better.

Current recipe in DL:
- large models
- large data
- large compute
- self-supervised learning
This is the recipe for foundation models.

i.i.d. learning - we have solved
But we have not solved the true learning problem (how humans learn):
- not i.i.d.
- sequential
- online (ChatGPT even doesn't solve it)
- continual

"Learning under non-stationarity" is a problem for NNs.

non-i.i.d. setting:
- catastrophic forgetting
- loss of plasticity (after some time NN stop learning)

LLMs are general purpose pre-trained initialization models.
But they are not AGI.
But for many practical situations - they are enough.

LLMs are a detour in our quest to understand the principles of learning.


### PyTorch Tutorial

Maithrraeye Srinivasan and Subho Pramenik, Amii

[Google Colab](https://colab.research.google.com/drive/14hyhJRQJol7ObzT9Iw2oJTDJ1E7boCkv#scrollTo=ctMpltukQsuP)
TODO Go over this colab

```python
torch.cuda.get_device_name(0)
```

PyTorch defaults to float32 for floats and int64 for integers for Tensors.
Convert using `.to()` or `.type()`.

Tensor ops:
- `.shape`
- `reshape()` vs `view()`
- `unsqueeze`, `squeeze`
- etc

Broadcasting

`requires_grad=True`

Visualization. TODO Try this!
```python
from torchviz import make_dot

make_dot(..)
```

TODO Methods of weights initialization in NNs.

`num_workers=0` - data loading is done in the main process

```python
torch.set_printopts(...)
```

TODO: Read about tips for reasonable LR for different architectures\

- data parallelism
- model parallelism
- hybrid approach

DistributedDataParallel in PyTorch
TODO Walk over the example from the notebook of the lecture

TODO What if GPUs are different? How does it handle this?


### Cohere, Communicating in Science

Jay Alammar

TBD

