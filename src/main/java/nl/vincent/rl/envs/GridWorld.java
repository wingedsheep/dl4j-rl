package nl.vincent.rl.envs;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;

public class GridWorld {
	
	double[][] rewards = new double[4][4];
	boolean[][] isFinal = new boolean[4][4];
	int currentX, currentY;
	boolean isDone;
	
	public enum Action {
		UP(0), DOWN(1), LEFT(2), RIGHT(3);
		
		int index;
		
		private Action(int index) {
			this.index = index;
		}
		
		public static Action getByIndex(int index) {
			return values()[index];
		}
		
		public int getIndex() {
			return index;
		}
	}
	
	/**
	 * -10 0 0 10
	 *   0 0 0  0
	 *   0 0 0  0
	 *   0 0 0 -10
	 */
	
	public GridWorld() {
		for (int i=0;i<rewards.length;i++) {
			for (int j=0;j<rewards[i].length;j++) {
				rewards[i][j] = -0.1;
			}
		}
		rewards[3][0] = -10;
		rewards[0][3] = -10;
		rewards[3][3] = 10;
		isFinal[3][0] = true;
		isFinal[0][3] = true;
		isFinal[3][3] = true;
		currentX = 0;
		currentY = 0;
	}
	
	public State reset() {
		currentX = 0;
		currentY = 0;
		isDone = false;
		return new State(new double[] {currentX, currentY});
	}
	
	public Observation step(Action action) {
		if (isDone) {
			System.err.println("Can't make a move when the env is finished");
		}
		State oldState = new State(new double[] {currentX, currentY});
		switch (action) {
			case UP: {
				if (currentY < 3) currentY ++ ;
				break;
			}
			case DOWN: {
				if (currentY > 0) currentY -- ;
				break;
			}
			case LEFT: {
				if (currentX > 0) currentX -- ;
				break;
			}
			case RIGHT: {
				if (currentX < 3) currentX ++ ;
				break;
			}
			default: break;	
		}
		State newState = new State(new double[] {currentX, currentY});
		double reward = rewards[currentX][currentY];
		boolean done = isFinal[currentX][currentY];
		isDone = done;
		return new Observation(oldState, action, newState, reward, done);
	}
}
