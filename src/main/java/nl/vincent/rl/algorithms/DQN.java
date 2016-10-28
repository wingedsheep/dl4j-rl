package nl.vincent.rl.algorithms;

import java.util.ArrayList;
import java.util.Random;

import nl.vincent.rl.common.Memory;
import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.GridWorld;
import nl.vincent.rl.envs.GridWorld.Action;
import nl.vincent.rl.networks.FullyConnected;

public class DQN {
	int inputSize;
	int outputSize;
	Memory memory;
	FullyConnected qModel1;
	FullyConnected qModel2;
	double discountFactor = 0.95;
	Random randomGen = new Random();

	public DQN() {
		inputSize = 2;
		outputSize = 4;
		memory = new Memory(100000);
		qModel1 = new FullyConnected(2, 4, new int[]{30, 30, 30}, 0.01);
		qModel2 = new FullyConnected(2, 4, new int[]{30, 30, 30}, 0.01);
	}
	
	public void run (int epochs, int steps) {
		GridWorld env = new GridWorld();
		double explorationRate = 1.0;
		for (int epoch = 0; epoch < epochs; epoch ++) {
			State state = env.reset();
			
			double totalReward = 0;
			int totalSteps = 0;
		    for (int t=0; t < steps; t++) {
		    	double[] qValues1 = qModel1.predict(state.getState());
		    	double[] qValues2 = qModel2.predict(state.getState());
		    	double[] qValues = new double[qValues1.length];
		    	for (int i=0; i < qValues1.length; i++) {
		    		qValues[i] = qValues1[i] + qValues2[i];
		    	}
		    	
		    	int action = selectAction(qValues, explorationRate);
		    	
		    	Observation obs = env.step(Action.getByIndex(action));
		    	
		    	totalReward += obs.getReward();
		    	totalSteps ++;
		    	
		    	memory.add(obs);
		    	
		    	state = obs.getState();
		    	
		    	learnOnMiniBatch(32);
		    	
		    	if (obs.isDone()) break;
		    }
		    System.out.println("Epoch "+epoch+" reward "+totalReward+" steps "+totalSteps);
		    explorationRate *= 0.995;
		}
	}
	
	private void learnOnMiniBatch(int size) {
		ArrayList<Observation> miniBatch = memory.getMiniBatch(size);
		ArrayList<double[]> inputs = new ArrayList<>();
		ArrayList<double[]> outputs = new ArrayList<>();
	
		double rand = randomGen.nextDouble();
		for (Observation obs : miniBatch) {
			double[] qValues = null;
			double[] qValuesNewState = null;
			if (rand < 0.5) {
				qValues = qModel1.predict(obs.getState().getState());
				qValuesNewState = qModel2.predict(obs.getNewState().getState());
			} else {
				qValues = qModel2.predict(obs.getState().getState());
				qValuesNewState = qModel1.predict(obs.getNewState().getState());
			}
			double targetValue = calculateTarget(qValuesNewState, obs.getReward(), obs.isDone());
			double[] input = obs.getState().getState();
			double[] output = qValues;
			output[obs.getAction().getIndex()] = targetValue;
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
		double random = randomGen.nextDouble();
		if (random < explorationRate) {
			int action = randomGen.nextInt(outputSize);
			return action;
		} else {
			int action = findMaxQIndex(qValues);
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
}
