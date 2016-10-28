package nl.vincent.rl.common;

import java.util.ArrayList;
import java.util.Random;

public class Memory {
	private int maxSize;
	private int currentPosition;
	private ArrayList<Observation> memories = new ArrayList<>();
	private Random rng = new Random();

	public Memory(int maxSize) {
        this.maxSize = maxSize;
        currentPosition = 0;
	}
	
	public void add(Observation obs) {
		memories.add(obs);
	}
	
	public ArrayList<Observation> getMiniBatch(int size) {
		if (size > memories.size()) {
			return memories;
		} else {
			ArrayList<Observation> batch = new ArrayList<>();
			for (int i =0 ; i< size; i++) {
				int index = rng.nextInt(memories.size());
				batch.add(memories.get(index));
			}
			return batch;
		}
	}
}
