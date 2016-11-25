package nl.vincent.rl.algorithms;

import nl.vincent.rl.common.ScoreAverageCounter;
import nl.vincent.rl.envs.Environment;
import nl.vincent.rl.envs.GridWorld;
import nl.vincent.rl.networks.FullyConnected;

public class A3C {
	private FullyConnected qModel;
	private FullyConnected targetModel;
	private int T;
	private int stepsPerTargetNetworkUpdate = 5000;
	ScoreAverageCounter scoreCounter = new ScoreAverageCounter(100);
	
	public A3C (Environment env, int[] hiddenLayers) {
		qModel = new FullyConnected(env.getStateSize(), env.getActionSize(), hiddenLayers, 0.01, FullyConnected.OuputType.LINEAR);
		targetModel = qModel.clone();
		T = 0;
	}
	
	public void run(int threads, int epochs, int steps) {	
		for (int i = 0 ; i< threads;i++) {
			Thread t = new Thread(new A3CThread(this, new GridWorld(), 10000, 100));
			t.start();
		}
	}
	
	public FullyConnected getModel() {
		return qModel;
	}
	
	public FullyConnected getTargetModel() {
		return targetModel;
	}
	
	public int getT() {
		return T;
	}
	
	public void increaseT() {
		T ++;
		synchronized (this) {
	    	if (T % getStepsPerTargetNetworkUpdate() == 0) {
	    		updateTargetNetwork();
	    	}
		}
	}
	
	public int getStepsPerTargetNetworkUpdate() {
		return stepsPerTargetNetworkUpdate;
	}

	public synchronized void updateTargetNetwork() {
		targetModel = qModel.clone();
	}

	public void addEpisodeReward(double totalReward) {
		scoreCounter.addScore(totalReward);
	    System.out.println("Average last 100 "+scoreCounter.getAverage());
	}
}
