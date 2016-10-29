package nl.vincent.rl.common;

public class Observation {
	private State state;
	private double reward;
	private boolean isFinal;
	
	public Observation(State state, double reward, boolean isFinal) {
		this.state = state;
		this.reward	= reward;
		this.isFinal = isFinal;
	}
	
	public State getState() {
		return state;
	}
	
	public double getReward() {
		return reward;
	}
	
	public boolean isFinal() {
		return isFinal;
	}
}
