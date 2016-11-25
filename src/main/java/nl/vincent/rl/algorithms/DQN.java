package nl.vincent.rl.algorithms;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import nl.vincent.rl.common.ExperienceReplayMemory;
import nl.vincent.rl.common.ExperienceReplayMemory.MemoryEntry;
import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.ScoreAverageCounter;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.Environment;
import nl.vincent.rl.networks.FullyConnected;

/**
 * Deep Q network (Dual)
 * @author vincentbons
 */
public class DQN {
	private int inputSize;
	private int outputSize;
	private ExperienceReplayMemory memory;
	private FullyConnected qModel1;
	private FullyConnected qModel2;
	private double discountFactor = 0.95;
	private int miniBatchSize = 32;
	private Random randomGen = new Random();
	private int[] hiddenLayers;
	private double startingExplorationRate;
	private double explorationRateDecay;
	private double learningRate;
	private Environment currentEnvironment;
	
	public DQN(DQNBuilder builder) {
		this.currentEnvironment = builder.environment;
		this.hiddenLayers = builder.hiddenLayers;
		this.learningRate = builder.learningRate;
		if (currentEnvironment != null) {
			initEnvironment(currentEnvironment);
		}
		this.memory = new ExperienceReplayMemory(builder.memorySize);
		this.discountFactor = builder.discountFactor;
		this.miniBatchSize = builder.miniBatchSize;
		this.startingExplorationRate = builder.startingExplorationRate;
		this.explorationRateDecay = builder.explorationRateDecay;
	}
	
	private void initEnvironment(Environment env) {
		this.currentEnvironment = env;
		this.inputSize = env.getStateSize();
		this.outputSize = env.getActionSize();
		qModel1 = new FullyConnected(inputSize, outputSize, hiddenLayers, learningRate, FullyConnected.OuputType.LINEAR);
		qModel2 = new FullyConnected(inputSize, outputSize, hiddenLayers, learningRate, FullyConnected.OuputType.LINEAR);
	}
	
	public void run(int epochs, int steps) {
		Environment env = currentEnvironment;
		double explorationRate = this.startingExplorationRate;
		ScoreAverageCounter scoreCounter = new ScoreAverageCounter(100);
		for (int epoch = 0; epoch < epochs; epoch ++) {
			State state = env.reset();
			
			double totalReward = 0;
			int totalSteps = 0;
		    for (int t=0; t < steps; t++) {
		    	double[] qValues1 = qModel1.predict(state.asList());
		    	double[] qValues2 = qModel2.predict(state.asList());
		    	double[] qValues = new double[qValues1.length];
		    	for (int i=0; i < qValues1.length; i++) {
		    		qValues[i] = qValues1[i] + qValues2[i];
		    	}
		    	
		    	int action = selectAction(qValues, explorationRate);
		    	
		    	Observation obs = env.step(action);
		    	
		    	totalReward += obs.getReward();
		    	totalSteps ++;
		    	
		    	memory.add(obs, state, action);
		    	
		    	state = obs.getState();
		    	
		    	learnOnMiniBatch(this.miniBatchSize);
		    	
		    	if (obs.isFinal()) break;
		    }
		    
		    scoreCounter.addScore(totalReward);
		    
		    System.out.println("Epoch "+epoch+" reward "+totalReward+" steps "+totalSteps+" Average last 100 "+scoreCounter.getAverage());
		    explorationRate *= this.explorationRateDecay;
		}
	}

	public void run (Environment env, int epochs, int steps) {
		initEnvironment(env);
		run (epochs, steps);
	}
	
	private void learnOnMiniBatch(int size) {
		MemoryEntry[] miniBatch = memory.getMiniBatch(size);
		ArrayList<double[]> inputs = new ArrayList<>();
		ArrayList<double[]> outputs = new ArrayList<>();
	
		double rand = randomGen.nextDouble();
		for (MemoryEntry entry : miniBatch) {
			double[] qValues = null;
			double[] qValuesNewState = null;
			if (rand < 0.5) {
				qValues = qModel1.predict(entry.getState().asList());
				qValuesNewState = qModel2.predict(entry.getNewState().asList());
			} else {
				qValues = qModel2.predict(entry.getState().asList());
				qValuesNewState = qModel1.predict(entry.getNewState().asList());
			}
			double targetValue = calculateTarget(qValuesNewState, entry.getReward(), entry.isFinal());
			double[] input = entry.getState().asList();
			double[] output = qValues;
			output[entry.getAction()] = targetValue;
			inputs.add(input);
			outputs.add(output);
		}
		
		if (rand < 0.5) {
			qModel1.train(inputs, outputs);
		} else {
			qModel2.train(inputs, outputs);
		}
	}
	
	private double calculateTarget(double[] qValuesNewState, double reward, boolean isDone) {
        if (isDone)
            return reward;
        else
            return reward + discountFactor * findMaxQ(qValuesNewState);
	}
	
	private int selectAction(double[] qValues, double explorationRate) {
		int[] availableActions =  currentEnvironment.getAvailableActions();
		
		double[] actionPickingQValues = qValues.clone();
		// Multiply selection probability by zero if action is not available.
		for (int i = 0; i < actionPickingQValues.length; i ++) {
			if(availableActions[i] == 0) {
				actionPickingQValues[i] = -1000;
			}
		}
		
		double random = randomGen.nextDouble();
		if (random < explorationRate) {
			int action = randomGen.nextInt(outputSize);
			while (availableActions[action] == 0) {
				action = randomGen.nextInt(outputSize);
			}
			return action;
		} else {
			int action = findMaxQIndex(actionPickingQValues);
			return action;
		}
	}
	
	private int findMaxQIndex(double[] qValues) {
		double max = qValues[0];
		int maxIndex = 0;

		for (int i = 1; i < qValues.length; i++) {
		    if (qValues[i] > max) {
		      max = qValues[i];
		      maxIndex = i;
		    }
		}
		return maxIndex;
	}
	
	private double findMaxQ(double[] qValues) {
		double max = qValues[0];

		for (int i = 1; i < qValues.length; i++) {
		    if (qValues[i] > max) {
		      max = qValues[i];
		    }
		}
		return max;
	}
	
	public void loadModel1(String fileName) throws IOException {
		qModel1.loadModel(fileName);
	}
	
	public void loadModel2(String fileName) throws IOException {
		qModel2.loadModel(fileName);
	}

	public static class DQNBuilder {
		private int memorySize = 100000;
		private int[] hiddenLayers = new int[] {30, 30, 30};
		private double discountFactor = 0.995;
		private double learningRate = 0.001;
		private Environment environment = null;
		private int miniBatchSize = 32;
		private double startingExplorationRate = 1.0;
		private double explorationRateDecay = 0.995;
		
		public DQNBuilder memorySize(int memorySize) {
			this.memorySize = memorySize;
			return this;
		}
		
		public DQNBuilder hiddenLayers(int[] hiddenLayers) {
			this.hiddenLayers = hiddenLayers;
			return this;
		}
		
		public DQNBuilder discountFactor(double discountFactor) {
			this.discountFactor = discountFactor;
			return this;
		}
		
		public DQNBuilder learningRate(double learningRate) {
			this.learningRate = learningRate;
			return this;
		}
		
		public DQNBuilder environment(Environment environment) {
			this.environment = environment;
			return this;
		}
		
		public DQNBuilder miniBatchSize(int miniBatchSize) {
			this.miniBatchSize = miniBatchSize;
			return this;
		}
		
		public DQNBuilder startingExplorationRate(double startingExplorationRate) {
			this.startingExplorationRate = startingExplorationRate;
			return this;
		}
		
		public DQNBuilder explorationRateDecay(double explorationRateDecay) {
			this.explorationRateDecay = explorationRateDecay;
			return this;
		}
		
		public DQN build() {
			return new DQN(this);
		}
	}
}
