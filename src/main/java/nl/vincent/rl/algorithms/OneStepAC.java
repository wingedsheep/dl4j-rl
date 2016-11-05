package nl.vincent.rl.algorithms;

import java.util.Random;

import org.deeplearning4j.nn.gradient.Gradient;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.Environment;
import nl.vincent.rl.networks.FullyConnected;

public class OneStepAC {
	FullyConnected policyModel;
	FullyConnected valueModel;
	int stateSize, actionSize;
	double alpha = 0.01;
	double beta = 0.01;
	Environment currentEnvironment;
	int[] hiddenLayersPolicy, hiddenLayersValue;
	double discountFactor;
	Random rng = new Random();
	
	public OneStepAC(Builder builder) {
		this.alpha = builder.alpha;
		this.beta = builder.beta;
		this.currentEnvironment = builder.environment;
		this.hiddenLayersPolicy = builder.hiddenLayersPolicy;
		this.hiddenLayersValue = builder.hiddenLayersValue;
		this.discountFactor = builder.discountFactor;
		if (currentEnvironment != null) {
			stateSize = currentEnvironment.getStateSize();
			actionSize = currentEnvironment.getActionSize();
			policyModel = new FullyConnected(stateSize, actionSize, hiddenLayersPolicy, 1, FullyConnected.OuputType.SOFTMAX);
			valueModel = new FullyConnected(stateSize, 1, hiddenLayersValue, 1, FullyConnected.OuputType.LINEAR);
		}
	}

	public void run (Environment env, int epochs, int steps) {
		initEnvironment(env);
		run (epochs, steps);
	}
	
	private void run (int epochs, int steps) {
		Environment env = currentEnvironment;
		for (int epoch = 0; epoch < epochs; epoch ++) {
			State state = currentEnvironment.reset();
			double totalReward = 0;
			int totalSteps = 0;
			for (int step = 0; step < steps; step ++) {
				int action = selectAction(state);
								
				Observation obs = env.step(action);
				
				double valuePredictionNewState = predictValue(obs.getState());
				double valuePredictionOldState = predictValue(state);
				double tdError;
				if (!obs.isFinal()) {
					tdError = obs.getReward() + discountFactor * valuePredictionNewState - valuePredictionOldState;
				} else {
					tdError = obs.getReward() - valuePredictionOldState;
				}
				
				double[] valueTarget = valueModel.predict(state.asList());
								
				// Zero because there is only one value for a state
				valueTarget[0] += 1;
				
				Gradient valueGradient = valueModel.gradient(state.asList(), valueTarget);
				valueGradient.gradient().muli(alpha * tdError);
				valueModel.applyGradient(valueGradient, 1);

				double[] policyTarget = policyModel.predict(state.asList());
				double actionProbability = policyTarget[action];
				policyTarget[action] += 1;
				Gradient policyGradient = policyModel.gradient(state.asList(), policyTarget);
				policyGradient.gradient().divi(actionProbability);
				policyGradient.gradient().muli(alpha * Math.pow(discountFactor, step) * tdError );
				policyModel.applyGradient(policyGradient, 1);
				
				if (isNaN(policyModel.predict(state.asList())[action])) {
					System.out.println("NaN output");
				}
				
				state = new State(obs.getState());
				
				totalReward += obs.getReward();
		    	totalSteps ++;
				
				if (obs.isFinal()) break;
			}
		    System.out.println("Epoch "+epoch+" reward "+totalReward+" steps "+totalSteps);
		}
	}
	
	boolean isNaN(double x){return x != x;}
	
	private double predictValue(State state) {
		return valueModel.predict(state.asList())[0];
	}
	
	private int selectAction(State state) {
		double[] policyOutput = policyModel.predict(state.asList());
		int[] availableActions =  currentEnvironment.getAvailableActions();
		
		// Multiply selection probability by zero if action is not available.
		double probSum = 0;
		for (int i = 0; i < policyOutput.length; i ++) {
			policyOutput[i] *= availableActions[i];
			probSum += policyOutput[i];
		}
		// Divide by probsum
		for (int i = 0; i < policyOutput.length; i ++) {
			policyOutput[i] /= probSum;
		}
		
		double random = rng.nextDouble();
		double cumulativeProbability = 0.0;
		
		for (int i = 0; i < policyOutput.length; i ++) {
			double probability = policyOutput[i];
			cumulativeProbability += probability;
			if (random < cumulativeProbability) {
				return i;
			}
		}
		
		// Not supposed to happen if the sum of policyOutput is 1.0
		return policyOutput.length - 1;
	}
	
	private void initEnvironment(Environment env) {
		currentEnvironment = env;
		stateSize = env.getStateSize();
		actionSize = env.getActionSize();
		policyModel = new FullyConnected(stateSize, actionSize, hiddenLayersPolicy, 0.01, FullyConnected.OuputType.SOFTMAX);
		valueModel = new FullyConnected(stateSize, 1, hiddenLayersValue, 0.01, FullyConnected.OuputType.LINEAR);
	}
	
	public static class Builder {
		double alpha = 0.01;
		double beta = 0.01;
		Environment environment;
		int[] hiddenLayersPolicy = new int[]{30, 30, 30}, hiddenLayersValue = new int[]{30, 30, 30};
		double discountFactor = 0.995;
		
		public Builder alpha(double alpha) {
			this.alpha = alpha;
			return this;
		}
		
		public Builder beta(double beta) {
			this.beta = beta;
			return this;
		}
		
		public Builder environment(Environment environment) {
			this.environment = environment;
			return this;
		}
		
		public Builder hiddenLayersPolicy(int[] hiddenLayersPolicy) {
			this.hiddenLayersPolicy = hiddenLayersPolicy;
			return this;
		}
		
		public Builder hiddenLayersValue(int[] hiddenLayersValue) {
			this.hiddenLayersValue = hiddenLayersValue;
			return this;
		}
		
		public Builder discountFactor(double discountFactor) {
			this.discountFactor = discountFactor;
			return this;
		}
		
		public OneStepAC build() {
			return new OneStepAC(this);
		}
	}
	
}
