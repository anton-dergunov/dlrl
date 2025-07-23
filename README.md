

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


### Implementing Deep Q-Networks (DQN) in PyTorch

Slides:
bit.ly/4f4joXO
https://docs.google.com/presentation/d/1Fi5Q0_oSg-mKNGFjJfMczJpFGX-AHFMHJtZIXPBFmLM/edit?pli=1

Code:
http://bit.ly/46mkMTs
[Google Colab](https://colab.research.google.com/drive/1YCWxmpBGv-wj0RYjYvyYFxDGaUdZ9QVy?usp=sharing)

Gymnasium library
MountainCar-v0: underpowered car needs to learn to roll back and forth to build momentum.

Deep Q-Network
Q*(s, a): expected total future reward

TODO Explain the loss function for Q-Learning algorithm

Key innovations:
- Experience replay buffer. TODO Why we need it? The reason that is it called off-policy algorithm. What is this?
- Target network
- $\epsilon$-greedy exploration

TODO What does `pip install package[strange_word]` does for strange word?

TODO * arguments in Python


### Cohere, Communicating in Science

Jay Alammar

TODO Model merging?

TODO Read paper on Command A in arxiv; and "Back to Basics: Revisiting REINFORCE..."
TODO RLOO youtube video (introduction to RL)

TODO Read "The Illustrated DeepSeek-R1" article

SWI bench

Youtube video "From Next Token Prediction to ..."

"Machine Learning Research Communication via Illustrated and Interactive Web Articles"

Hack: Write an easy explanation of a paper & send it to authors.

Wait for slides from Jay


### Startup panel

- "Rejection could be a career-defining event."
- "Show the messy, get feedback faster."
- "What can you build that people are willing to pay for - that's what matters."
- Recommendation for book "Crossing the chasm"


### What is a sequence language model?

Peter West

Open source model Tülu3

Originally: probability of text.
LM(x) = P(x) = P(w1, w2,.. wn)
Chain rule: P(w1) P(w2 | w1) P(w3 | w1, w2) * ... * P(wn | w1, ... wn-1)

Why not N-grams? Sparsity of language.
Papers "Infini-gram...", "OLMoTrace...", "AI as Humanity's Salieri..."

Want to evaluate: fit(LM, P)
$NLL = -sum_{x~P} \log LM(x)$
Perplexity: $PPL = e^{NLL}$

Basic N-gram models -> N-gram+++ (Kneser-Ney)

| Model           | PPL  |
| --------------- | ---- |
| KN 5gram        | 93.7 |
| Feed Forward NN | 85.1 |
| Recurrent NN    | 80.0 |
| 4xRNN + KN5     | 73.5 |

https://paperswithcode.com/sota/language-modelling-on-penn-treebank-word

Transformers and attention: test perplexity dropped a lot.

GPT-3 and few-shot -> post-training alignment -> InstructGPT

TODO Scaling laws for LLMs (how large dataset). How to spend the budget for better model perplexity?
TODO Pluralistic values? Different users have different needs.
TODO Add longer context without changing the model

Issues with allignment

Can LLMs generate one random number?
Aligned models capture human bias. Base models do better.

Can LLMs generate a sequence of random numbers?
Aligned models and base model do better.

Human bias: repetitions are bad.
Aligned models capture this bias. Base models do better.

Does this affect deeper behaviours?
"Write a 4-line poem about coffee" several times
Aligned - repetitive, most pleasant.
Base - different, most original.


### Intro to RL

Marlos C. Machado

RL is about learning from evaluative feedback rather than instructive.
A learning system that wants something, and that adapts its behaviour to get that.

```
NewEstimate <- OldEstimate + StepSize [Target - OldEstimate]
```

$Q_{n+1} \gets Q_n + \alpha [R_n - Q_n]$

Delayed credit assignment

Value functions are functions of states that estimate how good it is for the agnet to be in a given state.

Policy is a mapping of states to actions.

Monte Carlo -> Temporal difference:
$V(S_t) \gets V(S_t) ...$

TODO Temporal difference learning allow for boostrapping. What does it mean?

TODO Revise the formulas and examples

Q-learning: what is the best action that I could have done?

What makes RL problems unique
- Exploration
- Delayed credit assignment
- Generalization

Directly optimizing the policy being learned
Instead of learning value function - learn the policy directly

Policy gradient theorem

REINFORCE: Monte-Carlo Policy-Gradient Control


### Intro to Computer Vision

Ke Li

Vision is the richest sensory input, accounts for 83% of information captured with all senses.
50% of brain cells is devoted to vision processing.

Book "Foundations of Computer Vision" by Torralba et al
[Foundations of Computer Vision](https://visionbook.mit.edu/)

Loss of information: 3d -> 2d

Solutions:
1. Multiple views
2. Model all hypotheses

Spacial correlation?

Trick: subsample and use as input all subsampled images (so that kernels can work on various input sizes).

3d representations:
- Voxel Grids: cubic in the length of the smallest feature
- Polygonal Meshes: only represents the surface, quadratic in the length of the smallest feature
- Point Cloud

3D Gaussian Splatting (3DGS)

Proximity Attention Point Rendering (PAPR)


### Empirical AI Research

Adam White

Paper "Machine Learning that Matters"
Paper "Deep RL that Matters"

Papers:
- Crash course in statistics for RL:
    - A hitchhiker's guide to statistical comparisons of reinforcement learning algorithms
- Dealing with hyperparameters:
    - Evaluating the performance of reinforcement learning algorithms
- Insights from small scale experiments:
    - Revisiting rainbow: promoting more insightful and inclusive deep reinforcement learning research
- Smth else
    - Deep RL at the edge of the statistical precipice

Paper "The cross-environment hyperparameter setting benchmark for RL"

Do not use standard error.

Tolerance intervals

Key messages
- You need more runs/replications than you think
- Take a statistical points of view
- Watch out for untuned baselines
- Appeal to authority fallacy

Comprehensive guide to experiments in RL:
Paper "Empirical Design in Reinforcement Learning"

Hyperparameters in RL

Paper "A method for evaluating hyperparameters sensitivity in reinforcement learning"




