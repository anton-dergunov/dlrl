

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


### Generative AI for 3D Character Animation

Muhammad Gohar Javed

Usage in hospital: if any person moves in a different way than the normal person would.

Conditional signal for generative model: can be text or something else.

Representation of 3D motions: sequence of 3D poses. Each pose consists of a fixed kinematic tree, containing a fixed number of joints while connecting them with bones.

- Position or Velocity
- Orientation

Gen AI Techniques:
- GANs
- VAEs
- Diffusion Models, Autoregressive Transformers

Masked generative models

Motion tokens:
Continuous Latent Space (auto encoders)
Codebook?

Why discrete latent space?
Transformers work in discrete world, we want to reuse this. And we want to use with LLMs.

Paper "MoMask: Generative Masked Modeling of 3D Human Motions"
Residual Vector Quantized Variational AutoEncoder (VQ-VAE)

Explore MoMask for text2motion generation
https://ericguo5513.github.io/momask

Paper "InterMask: 3D Human Interaction Generation via Collaborative Masked Modeling"

Paper "Generative Human Motion Stylization in Latent Space" ICLR 2024

Future directions:
- root position
- other joints (hands) and facial emotions
- data acquisition
- physical plausibility

Paper "Go to Zero: Towards Zero-shot motion..."
Paper "ReinDiffuse: Crafting Physically Plausible motions with Reinforced Diffusion Model" WACV 2025


### RL in the Real World

Exploration remains an open question in RL.

Make sure agent can learn one goal first.

Performance degradation when going from offline to online.

Dealing with non-stationary.


### Model-Based RL

Martha White

Value iteration

Approximate dynamic programming

Model-based RL: How do we use RL when the agent needs to learn a model?

What are possible learned models?
Most obvious: p(s', r | s, a)

Dyna
Key idea: use RL updates on simulated experience

It is hard to beat experience replay.

Dyna-Q

Dyna = background planning

Important choices:
- The type of model
- Search-control (from which state to do planning?)

Simple example of Dyna: Experience Replay.

Non-parametric model in ML: do not learn parameters of the model.

Advantages of learned model over a transition buffer:
- Compactness
- Coverage
- Modularity
- Generalization

It can be had to beat ER.
Model-based RL remains less common than model-free (RL + ER).

Make sure models on agent state.
Model in latent space, not in observation space.

Expectation vs sample models.

Rollouts vs Temporal Abstraction
Errors will accumulate
Small errors in model accuracy can cause big errors in values

Temporal abstraction:
Define macro-actions. Directly model the outcome.
Instead of modelling all options, let the agent choose which to learn.
How to describe options to learn? Open unsolved problem.

Search control strategies: how to pick (s, a)

Other ways to use models


### Deep RL

Marlos C. Machado

https://cs231n.github.io/understanding-cnn/

Focus on generalization in this lecture. The agent can't visit every state-action combination.

RL were simply consumers of supervised learning.

Paper "The Arcade Learning Environment: An Evaluation Platform for General Agents"

Paper "Playing Atari with Deep RL"

RL algs. are unstable when using NNs for function approx.

Deep Q-Networks (DQN)
- Decorelates samples (and reuses them) with an experience replay buffer.
- Stabilizes the target it is regressing to using a target network.
- Lots of pre-processing and normalization to allow for the same hyperparameters.

Lots of details are in the appendix.

Prioritized experience replay
Double DQN
Dualy networks

Directly optimizing the policy instead of learning the value function:
- Learn how much we prefer an action over the other
- Define a stochastic policy

In partial observable situation stochastic policy is optimal (deterministic is not).

Value based methods: almost always off-policy.
Policy gradient methods:
- On-policy
- Off-policy

Off-policy: mismatch in data distribution between what you are learning.

PPO: on-policy.

The 37 Implementation Details of Proximal Policy Optimization
[The 37 Implementation Details of Proximal Policy Optimization · The ICLR Blog Track](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

Paper [The Phenomenon of Policy Churn](https://arxiv.org/abs/2206.00730)


### Inverse Problems

The number of equations is less than the number of unknowns.

**Optimization**:
argmin (y-f(x))^2 + S(x) (regularization)
**Bayesian perspective**:
argmin -log p(y | x) + -log p(x)

Activation functions as proximal mappings of regularization prior.

Why unroll neural networks?
- Data-limited generalization
- Interpretability
- Efficient architecture

Interpretable deep learning for deconvolutional analysis of neural signals

Generative models as learned priors

Solving partial differential equations with diffusion


### Transformers & Generative AI

finbarrtimbers@gmail.com
X: finbarrtimbers
https://finbarr.ca/transformers

Stable diffusion
ChatGPT

What made gen AI possible?
- The transformer
- Pre-training

Transformer is basically a large MLP. So, at lower levels of compute it does worse than specialized architectures, but better with more compute.

Scaling laws:
Larger models require fewer samples to reach the same performance.
Kaplan scaling vs Chinchilla scaling
TODO Check them

Attention:
- Full
- Chunked (thought to be partially responsible for Llama4's issues)
- Sliding window attention

Do we actually need multiple heads?
- Grouped-query (used in practice)
- Multi-query
They work almost as well

Normalization
Moved to pre-norm from post-norm LayerNorm
Pre-norm: there is a straight shot from the end to the beginning
Post-norm: gradient has gone through Nx LayerNorms, so shrinks/explodes

Frontiers of gen AI:
1. Running out of data -> RL
2. Context length (context doesn't really work past 100k tokens; no sparse attention mechanism that works)
3. Optimizers (Muon uses second order information)

Used to be an interview question in DeepMind: why don't we use second order optimizers?

RL with LLMs:
- RL on human feedback (RLHF)
- RL with verifiable rewards

Extremely under-explored; basic RL ideas have yet to be explored. Very little work on e.g. replay buffers.

OpenAI uses perplexity on their code base.
Can you reimplement Linux tools with a single prompt?


### Multi-Agent RL

Matt Taylor
https://irll.ca

Multi-agent systems:
- Homogeneous/heterogeneous
- Communicating?
- Cooperative / competitive / mixed

MARL:
- Controlling a single agent
- Could control a group of agent
- Could control all agents

OpenAI Hide and Seek

Current research topics:
- How much modeling is useful?
- Centralized/Decentralized Training & Execution; Centralized Training Decentralized Execution
- Where do rewards come from? (e.g. is crashing into a person worse than crash into a car?)
- What happens when 1+ of agents is a human?
- Mean Field RL (e.g. 10 closest cards)
- Knowledge sharing / transfer / teaching (e.g. one device teaches another)

http://marl-book.com
"Multi-Agent Reinforcement Learning. Foundations and Modern Approaches"
https://www.khoury.northeastern.edu/home/camato/tutorials.html
See other resources in the slides.


### Introduction to Design Sprint

Clay Lowe

5-day design sprint:
- Understand
- Define
- Sketch
- Prototype
- Test

Book "Sprint. Solve big problems and test new ideas in just five days"
https://www.thesprintbook.com

Team commitments:
- No distractions
- Respect the time frame and the breaks
- Engagement
- Work alone together


### The GenAI Paradigm in Healthcare

Using LLM (llama) for classification of CT reports. Tuned prompts.

Patient notes.

vLLM inference engine
[GitHub - vllm-project/vllm: A high-throughput and memory-efficient inference and serving engine for LLMs](https://github.com/vllm-project/vllm)

WhisperX

Llama-3-OpenBioLLM-70B

OS Jenkins AI Scribe
PHAIRlab@ualberta.ca

Foundation models for human body scans
How can we possibly label such a vast amount of data? RLHF


### AI Ethics

Geoffrey Rockwell
University of Alberta
www.geoffreyrockwell.com

Montreal Declaration for Responsible AI
https://www.montrealdeclaration-responsibleai.com

https://www.rolls-royce.com/innovation/the-aletheia-framework.aspx
https://learn.microsoft.com/en-us/azure/architecture/guide/responsible-innovation/judgementcall

Businesses prefer to work with companies that are not a part of a scandal.
Ethics does not make you money. Ethics saves you money when the time comes.


### Debugging foundation models: the elephant in the room

Randy Goebel

GwenAI
Manus
Grok
DeepSeek

Silver et al. Reward is enough 2021

https://www.concentrix.com/insights/blog/a-guide-to-the-evolution-of-ai/

ASI = artificial superintelligence

Gary Marcus blog on Midjourney picture of a man and unicorn.

Neurosymbolic foundational models

From Statistical Relational to Neuro-symbolic Artificial Intelligence
https://arxiv.org/abs/2003.08316

https://adamfard.com/blog/explainable-ai

Model Editing
https://arxiv.org/abs/2406.19354v1

https://www.opemindresearch.org


### Richard Sutton keynote. Welcome to the Era of Experience

How do you notice intelligence? Purpose.

Experience is the data you get during normal life.

RL is powerful, because requires so little from people.
- no labeled examples
- ...

Multiple objectives in RL. But reward is the ultimate goal. But to get reward, it is important to get other things.

We are in the era of human data. Systems trained to predict humans' next words, not to predict or control the word.
We need to enter the era of experience.
Silver & Sutton "Welcome to the era of experience" (2025)

Cooperation is humanity's super-power.
It is good for economies to have agents with different goals.

How to feel about AI?
- The age of particles
- The age of stars
- The age of ~~life~~ replicators
- The age of ~~machines~~ design

Biological things are replicated. Technological things are created and they are easier to improve.

What would it mean to take design all the way?
Design things that are themselves capable of designing. This is what we do with AI.

In addition to learning weights, we should learn step sizes. (Step size adaptation paper.)

TODO See reading recommendation on Richard Sutton's web page.


### Panel

Sutten-Burreton model of classical conditioning? As temporal-difference learning

Physics discoveries with ML (e.g. black hole): they included prior information.

AI Scientist


### Designing scalable and efficient NNs

TODO Apply for RBC Borealis Research Internship Program. Deadline: 7 September
[ML Research Internships - RBC Borealis](https://rbcborealis.com/internships/)

Conditional Neural Processes

Recurrent formulation of attention

Rise of linear RNNs, linear attention, state-space models
Parallel Scan algorithm (Blelloch, 1990)

Methods:
1. Attention as a recurrent NN (how?)
2. Simplifying RNNs (drop hidden state dependencies: LSTM -> minLSTM)

The idea is to use retrieval for cross attention: O(N) -> O(log(N))
k-d-tree; use RL to search

Mamba: bias for more recent tokens.


### MLOps

Apache Airflow
mlflow

"dvc pull"?
"supervisorctl restart process"?

DevOps considerations:
- volume of use
- acceptable latency
- data causality
- geo availability
- SLA: failover, uptime constraints
- pre-processing (features)
- model & post processing privacy
- service provider affinity (Google, AWS, Azure)
- budget

Shadeform

[GitHub - casey/just: 🤖 Just a command runner](https://github.com/casey/just)

ngrok
[ngrok Agent CLI Quickstart \| ngrok documentation](https://ngrok.com/docs/getting-started/)

Gradio

HuggingFace Spaces
ZeroGPU - free tier, might get or not

Docker + FastAPI
Use with HF Spaces as well

Benchmarking and stress testing:
Locust (`pip install locust`)
(Simulates users accessing the API)


### Search

Sokoban
Single-agent pathfinding

Policy-guided heuristic search
Three parts:
1. Searching with a polity
2. Searching with a policy and heuristic function
3. Learning a policy

Policy: probability of finding a solution
Best-first search with probabilities

P(n) = P(n') * P(n|n')
Base: P(n0) = 1
Search algorithm lacks completeness.

Levin cost:
d(n) / P(n) where d(n) is the depth of the tree, d(root) = 0
d(n1) / P(n1) = 1/0.2 = 5
d(n2) / P(n2) = 1/0.8 = 1.25

Main property: the Levin cost of the goal plus 1 is an upper bound on the number of expansions.
If we learn policies that minimize levin cost, then we minimize the search effect.

Property 1: LevinTS performs the search in best-first order.
Property 2: The sum of probabilities of leaf nodes of a tree is 1.

Can we do better?

Policy-guided heuristic search (PHS)

PHS*

Solution: context moels
Use Levin cost -> shrinking the size of the tree

If you need to solve combinatorial search problems, the Levin family should be your go-to option. Not MCTS.


### Introduction to Optimization

Mark Schmidt
UBC

https://www.cs.ubc.ca/~schmidtm/Courses/5XX-S22

Gradient descent: Cauchy 1840s

SGD converges with decreasing a_k.

Polyak-Lojasiewicz Inequality and Invexity

Non-linear conjugate gradient

Question 1: How do I set the step size?

GD:
L (Lipschitz constant): how fat the gradient can change.
a_k vs 2/L
TODO: Run experiments that show this.
Armijo Backtracking
Malitsky-Mischenko
Barzilai-Borwein
Do not work for SGD in general

SGD:
$\sigma_k^2$ - variation in the gradients ("noise")
size of gradient of overall function

Two phases:
- gradient dominated
- noise dominated

How to adjust the SGD step size?
Most methods are bad.

Weird dynamics:
- Edge of stability
- Initialization and warm-up
- Catapults

Question 2: How to pick batch size?

GPT-3: 3 million batch size
Google: batch size of millions
Bigger batches lead to a longer gradient-dominated phase

Gradually growing batch sizes

Question 3: In what order should we process data?

Random shuffling
Importance sampling
Curriculum learning (easy -> hard, but hard to define)
  Near the end of training, use higher-quality or task-specific data

Question 4: Are there faster algorithms than SGD?

We have faster algorithms than SD (heavy-ball, conjugate gradient a momentum term, Nesterov's accelerated gradient, Newton's method)
Faster than SGD? I depends:
- In gradient-dominated phrase: yes
- In noise-dominated phase: do not lead to faster convergence

There are faster algorithms for the noise-dominated phrase: varience-reduced SGD (SAG and SVRG). But does not help for deep learning.

Question 7: Should I just use Adam with default parameters?

Algoperf: Adam wins the competition.

For small batch sizes, Adam does not help.
For large batch size, gap between Adam and SGD grows.

Adam make similar progress on all labels.
Adam outperforms SGD on vision datasets with heavy-tailed labels.

Speedrunning and Muon
Gradient of the step sizes


### Robotics

Dr. Jun Jin

Embodied AI



