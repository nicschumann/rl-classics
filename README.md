# RL Classics

This repository is a playground in which I'm implementing a few different RL algorithms that I consider "classic," based on my limited experience of the field. The repo is intended as a pedagogical execise to help me learn a bit more about how these different algorithms work. I started messing about in this repo as part of my 2021 "summer school" on simulation and reinforcement learning (which I organized for three talented students from RISD). For that reason, it's also partially intended as supplementary material for our units on value function approximation, policy gradients, and model-based RL.

The implementations try to stay as straight-forward and as close to the math as possible. (My implementations were derived while watching David Silver's 2015 RL Course, [which is available on YouTube](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZBiG_XpjnPrSNw-1XQaM_gB).) Additionally, each implementation is in the state it was when I first got it working. There's a lot opf opportunity for refactoring in these files, but refactoring is the product of insight into your problem, and compressing your solution into a more reusable form. In this case, I want to value ease of reading the source code above refactoring into more flexible representations.

## VFA (`vfa.py`)

*TD(0) Q-learning + Replay Memory + Epsilon-Greedy Policy*

I implemented a value function approximator first. The action-value function (often called Q) is approximated by a shallow and small neural network. An exponentially decaying epsilon schedule is selected to guide exploration. The learning rate alpha is decayed by a factor of 0.9 every 100 episodes. A fixed discount factor for the MDP is set to 0.95, and a fixed batch size of 64 samples for the replay memory is used. Under favorable seeds, the VFA can learn an optimal policy (keep the pole balanced for 500 timesteps) for `CartPole-v1` in about 400 episodes.


## PGAC (`acpg.py`)

*Actor-Critic Policy Gradient + Replay Memory*

Both the policy (Actor) and the state-value function (Critic, often called V) are parametrized by neural networks. The state value network learns a mapping from the state space to a single score representing the expected future return from the state. It is trained by SGD on a TD(0) target. The policy network learns a softmax policy by ascending the gradient of the log probability of each action it takes, weighted by the advantage of that action (the Q-function minus the V-function). The advantage function is not explicitly computed, but the TD(0) error is intepreted as a biased sample of the advantage and used in its place.


Two separate learning rates are used for the v-function and the policy. Each are decayed at slightly different rates per 100 episodes. A fixed MDP discount factor of .997 and a fixed batch size of 128 are used. Under favorable seeds, the ACPG can learn an optimal policy (keep the pole balanced for 500 timesteps) for `CartPole-v1` in about 100 episodes.
