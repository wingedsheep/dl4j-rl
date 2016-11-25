package nl.vincent.rl.algorithms;

import java.util.Random;

import org.deeplearning4j.nn.gradient.Gradient;
import org.nd4j.linalg.api.ndarray.INDArray;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.Environment;

public class A3CThread implements Runnable {
	private A3C global;
	private int epochs, steps;
	private int counter;
	private int stepsPerGlobalUpdate = 2;
	private double learningRate = 0.001;
	private INDArray cumulatedGradient;
	private Environment env;
	private Random randomGen = new Random();
	double discountFactor = 0.99;
	
	public A3CThread(A3C global, Environment env, int epochs, int steps) {
		this.env = env;
		this.global = global;
		this.epochs = epochs;
		this.steps = steps;
	}

	@Override
	public void run() {
		counter = 0;
		// Get the current gradient, which should be zero
		State initState = env.reset();
		global.getModel().gradient(initState.asList(), global.getModel().predict(initState.asList()));
		cumulatedGradient = global.getModel().getModel().gradient().gradient();
		cumulatedGradient.muli(0);
		double explorationRate = 1.0d;
		double explorationRateDecay = 0.99d;
		
		for (int epoch = 0; epoch < epochs; epoch ++) {
			State state = env.reset();
			double totalReward = 0;
			
		    for (int t=0; t < steps; t++) {
		    	double[] qValues = global.getModel().predict(state.asList());
		    	
		    	int action = selectAction(qValues, explorationRate);
		    	
		    	Observation obs = env.step(action);
		    	
		    	totalReward += obs.getReward();
		    	
		    	double y = calculateTarget(obs);
		    	qValues[action] = y;
		    	Gradient grad = global.getModel().gradient(state.asList(), qValues);
		    	cumulatedGradient.addi(grad.gradient());
		    	
		    	state = obs.getState();
		    	
		    	t++;
		    	counter ++;
		    	global.increaseT();
		    	if (counter % stepsPerGlobalUpdate == 0) {
		    		synchronized (global) {
//		    			cumulatedGradient.divi(stepsPerGlobalUpdate);
		    			cumulatedGradient.muli(learningRate);
			    		global.getModel().applyGradient(cumulatedGradient);
			    		cumulatedGradient.muli(0);
		    		}
		    	}
		    	if (obs.isFinal()) break;
		    }
		    global.addEpisodeReward(totalReward);
		    epoch ++;
		    explorationRate *= explorationRateDecay;
		    explorationRate = Math.max(explorationRate, 0.01);
		}
	}
	
	private double calculateTarget(Observation obs) {
		double[] qValuesNewState = global.getTargetModel().predict(obs.getState().asList());
        if (obs.isFinal())
            return obs.getReward();
        else
            return obs.getReward() + discountFactor * findMaxQ(qValuesNewState);
	}
	
	private int selectAction(double[] qValues, double explorationRate) {
		int[] availableActions =  env.getAvailableActions();
		
		double[] actionPickingQValues = qValues.clone();
		// Multiply selection probability by zero if action is not available.
		for (int i = 0; i < actionPickingQValues.length; i ++) {
			if(availableActions[i] == 0) {
				actionPickingQValues[i] = -1000;
			}
		}
		
		double random = randomGen.nextDouble();
		if (random < explorationRate) {
			int action = randomGen.nextInt(env.getActionSize());
			while (availableActions[action] == 0) {
				action = randomGen.nextInt(env.getActionSize());
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
}
