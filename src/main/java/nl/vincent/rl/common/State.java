package nl.vincent.rl.common;

public class State {
	private double[] state;
	
	public State(double[] state) {
		this.state = state;
	}
	
	public State(State source) {
		this.state = source.state.clone();
	}
	
	public double[] asList() {
		return state;
	}
}
