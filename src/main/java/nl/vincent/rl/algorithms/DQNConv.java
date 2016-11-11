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
import nl.vincent.rl.networks.Convolutional;
import nl.vincent.rl.networks.Convolutional.OuputType;

/**
 * Deep Q network (Dual)
 * @author vincentbons
 */
public class DQNConv {
	private int inputSize;
	private int outputSize;
	private ExperienceReplayMemory memory;
	private Convolutional qModel1;
	private Convolutional qModel2;
	private double discountFactor = 0.95;
	private int miniBatchSize = 32;
	private Random randomGen = new Random();
	private int[] hiddenLayers;
	private int[] filterSizes = new int[] {8, 4, 3};
	private int[] strides = new int[] {2, 1, 1};
	private int[][] paddings;
	private int width, height, depth;
	private double startingExplorationRate;
	private double explorationRateDecay;
	private double learningRate;
	private Environment currentEnvironment;
	private int[] fullyConnectedLayers;
	
	public DQNConv(DQNBuilder builder) {
		this.currentEnvironment = builder.environment;
		this.hiddenLayers = builder.hiddenLayers;
		this.learningRate = builder.learningRate;
		this.filterSizes = builder.filterSizes;
		this.strides = builder.strides;
		this.paddings = builder.paddings;
		this.width = builder.width;
		this.height = builder.height;
		this.depth = builder.depth;
		this.fullyConnectedLayers = builder.fullyConnectedLayers;
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
		qModel1 = new Convolutional(width, height, depth, outputSize, hiddenLayers, fullyConnectedLayers, filterSizes, strides, paddings, learningRate, OuputType.LINEAR);
		qModel2 = new Convolutional(width, height, depth, outputSize, hiddenLayers, fullyConnectedLayers, filterSizes, strides, paddings, learningRate, OuputType.LINEAR);
	}
	
	public void run(int epochs, int steps) {
		Environment env = currentEnvironment;
		double explorationRate = this.startingExplorationRate;
		ScoreAverageCounter scoreCounter = new ScoreAverageCounter(100);
		for (int epoch = 0; epoch < epochs; epoch ++) {
			State state = env.reset();
			
			if (epoch % 1000 == 0 && epoch != 0) {
				try {
					qModel1.saveModel("qModel1_"+epoch+".zip");
					qModel2.saveModel("qModel2_"+epoch+".zip");
				} catch (Exception e) {}
			}
			
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
		private int[] hiddenLayers = new int[] {30, 30};
		private int[] filterSizes = new int[] {8, 4, 3};
		private int[] strides = new int[] {2, 1, 1};
		private int[][] paddings = new int[][] {};
		private int width, height, depth;
		private int[] fullyConnectedLayers;
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
		
		public DQNBuilder setFilterSizes(int[] filterSizes) {
			this.filterSizes = filterSizes;
			return this;
		}
		
		public DQNBuilder setStrides(int[] strides) {
			this.strides = strides;
			return this;
		}
		
		public DQNBuilder setPaddings(int[][] is) {
			this.paddings = is;
			return this;
		}
		
		public DQNBuilder setWidth(int width) {
			this.width = width;
			return this;
		}
		
		public DQNBuilder setHeight(int height) {
			this.height = height;
			return this;
		}
		
		public DQNBuilder setDepth(int depth) {
			this.depth = depth;
			return this;
		}
		
		public DQNBuilder setFullyConnectedLayers(int[] fullyConnectedLayers) {
			this.fullyConnectedLayers = fullyConnectedLayers;
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
		
		public DQNConv build() {
			return new DQNConv(this);
		}
	}
}
