# Various machine learning scripts and tools in Python


- ***neural_networks.py***: Implementation of scalable Multilayer Perceptron (MLP) and Radial Basis Function network (RBF) with numpy only.
- ***pendulum.py***: A simple pendulum to be controlled by a torque at its hinge.
- ***looptools.py***: Contains a context manager which allows the SIGINT signal to be processed asynchronously and a class to plot variables incrementally.
- ***CACLA_pendulum.py***: Implementation of the Continuous Actor Critic Learning Automaton (CACLA) [1] to balance a pendulum.
- ***DDPG_vanilla.py***: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm [1] with TensorFlow.
- ***DDPG_PER.py***: Implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm [2] with TensorFlow and enhanced with Prioritized Experience Replay (PER) [3].

[1] Van Hasselt, Hado, and Marco A. Wiering. "Reinforcement learning in continuous action spaces."<br />
    2007 IEEE International Symposium on Approximate Dynamic Programming and Reinforcement Learning. IEEE, 2007.<br />
[2] Lillicrap, Timothy P., et al. "Continuous control with deep reinforcement learning." arXiv preprint arXiv:1509.02971 (2015).<br />
[3] Schaul, Tom, et al. "Prioritized experience replay." arXiv preprint arXiv:1511.05952 (2015).
