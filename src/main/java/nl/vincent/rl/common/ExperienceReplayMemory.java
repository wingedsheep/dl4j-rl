package nl.vincent.rl.common;

import java.util.Random;

public class ExperienceReplayMemory {
	private int maxSize;
	private int currentPosition = 0;
	private int currentSize = 0;
	private MemoryEntry[] memories;
	private Random rng = new Random();
	
	public class MemoryEntry {
		private State state;
		private int action;
		private double reward;
		private State newState;
		private boolean isFinal;
		
		public MemoryEntry(State state, int action, Observation observation) {
			this.state = new State(state);
			this.action = action;
			this.reward = observation.getReward();
			this.newState = new State(observation.getState());
			this.isFinal = observation.isFinal();
		}
		
		public int getAction() {
			return action;
		}
		
		public State getNewState() {
			return newState;
		}
		
		public double getReward() {
			return reward;
		}
		
		public State getState() {
			return state;
		}
		
		public boolean isFinal() {
			return isFinal;
		}
	}

	public ExperienceReplayMemory(int maxSize) {
        this.maxSize = maxSize;        
        memories = new MemoryEntry[maxSize];
	}
	
	public void add(Observation obs, State currentState, int action) {
		MemoryEntry newMemory = new MemoryEntry(currentState, action, obs);
		memories[currentPosition] = newMemory;
		currentPosition ++;
		if (currentSize < maxSize) {
			currentSize ++;
		}
		if (currentPosition >= maxSize) {
			currentPosition = 0;
		}
	}
	
	public MemoryEntry[] getMiniBatch(int size) {
		if (size > currentSize) {
			MemoryEntry[] miniBatch = new MemoryEntry[currentSize];
			for (int i = 0; i < currentSize; i++) {
				miniBatch[i] = memories[i];
			}
			return miniBatch;
		} else {
			MemoryEntry[] batch = new MemoryEntry[size];
			for (int i =0 ; i< size; i++) {
				int index = rng.nextInt(currentSize);
				batch[i] = memories[index];
			}
			return batch;
		}
	}
}
