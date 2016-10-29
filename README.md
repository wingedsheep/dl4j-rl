# dl4j-rl

This project will be home to my reinforcement learning experiments using dl4j.
So far there is a toy test case for DQN in a gridworld.

# examples


```java
// Create a DQN solver (hidden layers have ReLu activation and output layer has softmax activation)
DQN dqn = new DQN.DQNBuilder()
		.discountFactor(0.995)
		.explorationRateDecay(0.995)
		.hiddenLayers(new int[] {30, 30, 30})
		.memorySize(100000)
		.miniBatchSize(32)
		.learningRate(0.01)
		.startingExplorationRate(1.0).build();

// Try to solve the gridworld using DQN (10000 epochs, max 200 steps per epoch)
dqn.run(new GridWorld(), 10000, 200);
```

Thanks to https://github.com/OrkoHunter/Minesweeper/tree/master/dist for the minesweeper env.