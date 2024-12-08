# Asynchronous-Methods-for-Deep-Reinforcement-Learning

The project proposes a conceptually simple and lightweight framework for deep reinforcement learning that uses asynchronous gradient descent for optimization of deep neural network controllers. This is the implementation of the research paper with the same name available [here](https://arxiv.org/pdf/1602.01783.pdf). Flappy bird game is used for the implementation.

## Installation Dependencies:

- Python 3
- Tensorflow 
- pygame
- OpenCV-Python

## How to run?

- First clone the project into your local system
```
git clone https://github.com/Nuclearstar/Asynchronous-Methods-for-Deep-Reinforcement-Learning.git
```
- Then change directory to this project
```
cd Asynchronous-Methods-for-Deep-Reinforcement-Learning
```
- Then setup a virtual env
```
python -m venv myenv
```
- Then activate your virtual env
```
cd myenv
cd Scripts
activate
```
- Further change directory to project root
```
cd ..
cd ..
```
- Next install all the required packages in the virtual env
```
pip install -r requirements.txt
```
- Now you are ready to run the program
```
python deep_q_network_actual.py
```

## Objectives:

We present asynchronous variants of three standard reinforcement learning algorithms:
1. One-step method
2. Actor critic method
3. n steps Q-learning method.

## Proposed solution methods

- We present multi-threaded asynchronous n-step Q-learning and advantage actor-critic.

- First, we use asynchronous actor-learners, similarly to the Gorila framework, but instead of using separate machines and a parameter server, we use multiple CPU threads on a single machine.

- Second, we make the observation that multiple actors-learners running in parallel.

## System design

### 1. Deep Q-Network

- Deep Q-Network is a convolutional neural network, trained with a variant of Q-learning.

- Q-function with a neural network, that takes the state and action as input and outputs the corresponding Q-value.

- One forward pass through the network and having all Q-values for all actions available is made use of.
![alt text](https://github.com/Nuclearstar/Asynchronous-Methods-for-Deep-Reinforcement-Learning/blob/master/images/q.JPG)

### 2. Environment

![alt text](https://github.com/Nuclearstar/Asynchronous-Methods-for-Deep-Reinforcement-Learning/blob/master/images/preprocess.png)

### 3. Architecture of the network

![alt text](https://github.com/Nuclearstar/Asynchronous-Methods-for-Deep-Reinforcement-Learning/blob/master/images/network.png)

## Implementation

Flappy bird game is used for the implementation.

![alt text](https://github.com/Nuclearstar/Asynchronous-Methods-for-Deep-Reinforcement-Learning/blob/master/images/flappy_bird_demp.gif)

## Conclusion and future work 

- We have presented asynchronous versions of three standard reinforcement learning algorithms and also showed that they are able to train neural network controllers on a variety of domains in a stable manner. 

- Combining other existing reinforcement learning methods or recent advances in deep reinforcement learning with our asynchronous framework presents many possibilities for immediate improvements to the methods we presented.
