package nl.vincent.rl.envs;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;

public abstract class Environment {
	protected int actions;
	protected int stateSize;
	
	public Environment() {}
	
	public Environment(int actions, int stateSize) {
		this.actions = actions;
		this.stateSize = stateSize;
	}
	
	public abstract State reset();
	public abstract Observation step(int action);
	public int getStateSize() {
		return stateSize;
	}
	public int getActionSize() {
		return this.actions;
	}
	public int[] getAvailableActions() {
		int[] availableActions = new int[actions];
		for (int i=0;i<availableActions.length;i++) {
			availableActions[i] = 1;
		}
		return availableActions;
	}
}
