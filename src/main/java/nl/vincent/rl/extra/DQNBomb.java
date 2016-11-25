package nl.vincent.rl.extra;

import java.io.IOException;

import nl.vincent.rl.common.ExperienceReplayMemory;
import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.ScoreAverageCounter;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.Environment;
import nl.vincent.rl.networks.FullyConnected;

/**
 * Deep Q network (Dual)
 * @author vincentbons
 */
public class DQNBomb {
	private int inputSize;
	private ExperienceReplayMemory memory;
	private FullyConnected qModel1;
	private FullyConnected qModel2;
	private FullyConnected bombProbabilityPredictor;
	private int[] hiddenLayers;
	private double learningRate;
	private Environment currentEnvironment;
	
	public DQNBomb(Builder builder) {
		this.currentEnvironment = builder.environment;
		this.hiddenLayers = builder.hiddenLayers;
		this.learningRate = builder.learningRate;
		if (currentEnvironment != null) {
			initEnvironment(currentEnvironment);
		}
		this.memory = new ExperienceReplayMemory(builder.memorySize);
	}
	
	private void initEnvironment(Environment env) {
		this.currentEnvironment = env;
		this.inputSize = env.getStateSize();
		bombProbabilityPredictor = new FullyConnected(25, 1, hiddenLayers, learningRate, FullyConnected.OuputType.LINEAR);
	}
	
	public void run(int epochs, int steps) {
		Environment env = currentEnvironment;
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
		    	
		    	int action = selectActionBasedOnBombProbabilities(state);
		    	
		    	Observation obs = env.step(action);
		    	
		    	totalReward += obs.getReward();
		    	totalSteps ++;
		    	
		    	memory.add(obs, state, action);
		    	
		    	learnBombProbability(state, action, obs.getReward());
		    	
		    	state = obs.getState();		    	
		    	
		    	if (obs.isFinal()) break;
		    }
		    
		    scoreCounter.addScore(totalReward);
		    
		    System.out.println("Epoch "+epoch+" reward "+totalReward+" steps "+totalSteps+" Average last 100 "+scoreCounter.getAverage());
		}
	}

	public void run (Environment env, int epochs, int steps) {
		initEnvironment(env);
		run (epochs, steps);
	}
	
	private void learnBombProbability(State state, int action, double reward) {
		// Bomb detection kernel
		
		// row above
		double[] bombKernelInput = getSurroundingsOfAction(action, state);
		
		double[] bombKernelOutput = new double[1];
		bombKernelOutput[0] = reward < 0 ? 1.0 : 0.0;
		
		bombProbabilityPredictor.train(bombKernelInput, bombKernelOutput);
	}
	
	private double[] getSurroundingsOfAction(int action, State state) {
		// Bomb detection kernel
		int boardSize =  (int) Math.sqrt(inputSize);
		int actionY = action / boardSize;
		int actionX = action - (actionY * boardSize);
		
		int i = 0;
		double[] bombKernelInputX = new double[25];
		for (int y = actionY - 1; y <= actionY +1 ; y ++) {
			for (int x = actionX - 1; x <= actionX +1 ; x ++) {
				if (fromXYToAction(x, y, boardSize) != -1) {
					bombKernelInputX[i] = state.asList()[fromXYToAction(x, y, boardSize)];
				} else {
					bombKernelInputX[i] = -2;
				}
				i++;
			}
		}

		return bombKernelInputX;
	}
	
	private int fromXYToAction(int x, int y, int boardSize) {
		if (x < 0 || x >= boardSize) return -1;
		if (y < 0 || y >= boardSize) return -1;
		return x + y * boardSize;
	}
	
	private int selectActionBasedOnBombProbabilities(State state) {
		int[] availableActions =  currentEnvironment.getAvailableActions();

		double min = 100000.0;
		int minAction = availableActions[0];
		int action = 0;
		for (int isAvailable : availableActions) {
			if (isAvailable == 1) {
				double[] surroundings = getSurroundingsOfAction(action, state);
				double bombProbability = bombProbabilityPredictor.predict(surroundings)[0];
				if (bombProbability < min) {
					min = bombProbability;
					minAction = action;
				}
			}
			action ++;
		}
		return minAction;
	}
	
	public void loadModel1(String fileName) throws IOException {
		qModel1.loadModel(fileName);
	}
	
	public void loadModel2(String fileName) throws IOException {
		qModel2.loadModel(fileName);
	}

	public static class Builder {
		private int memorySize = 100000;
		private int[] hiddenLayers = new int[] {30, 30, 30};
		private double learningRate = 0.001;
		private Environment environment = null;
		
		public Builder memorySize(int memorySize) {
			this.memorySize = memorySize;
			return this;
		}
		
		public Builder hiddenLayers(int[] hiddenLayers) {
			this.hiddenLayers = hiddenLayers;
			return this;
		}
		
		public Builder learningRate(double learningRate) {
			this.learningRate = learningRate;
			return this;
		}
		
		public Builder environment(Environment environment) {
			this.environment = environment;
			return this;
		}
		
		public DQNBomb build() {
			return new DQNBomb(this);
		}
	}
}
