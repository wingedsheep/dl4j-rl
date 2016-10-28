package nl.vincent.rl.common;

import nl.vincent.rl.envs.GridWorld.Action;

public class Observation {
	private State state;
	private Action action;
	private State newState;
	private double reward;
	private boolean done;
	
	public Observation(State state, Action action, State newState, double reward, boolean done) {
		this.state = state;
		this.action = action;
		this.newState = newState;
		this.reward	= reward;
		this.done = done;
	}
	
	public State getState() {
		return state;
	}
	
	public double getReward() {
		return reward;
	}
	
	public boolean isDone() {
		return done;
	}
	
	public State getNewState() {
		return newState;
	}
	
	public Action getAction() {
		return action;
	}
}
