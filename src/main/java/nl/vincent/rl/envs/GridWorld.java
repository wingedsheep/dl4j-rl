package nl.vincent.rl.envs;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;

public class GridWorld extends Environment {
	double[][] rewardFields = new double[4][4];
	boolean[][] finalFields = new boolean[4][4];
	int currentX, currentY;
	boolean isDone;
	
	// Actions
	public final static int UP = 0;
	public final static int DOWN = 1;
	public final static int LEFT = 2;
	public final static int RIGHT = 3;
	
	/**
	 * -10 0 0 10
	 *   0 0 0  0
	 *   0 0 0  0
	 *   0 0 0 -10
	 */
	
	public GridWorld() {
		super(4, 2);
		for (int i=0;i<rewardFields.length;i++) {
			for (int j=0;j<rewardFields[i].length;j++) {
				rewardFields[i][j] = -0.1;
			}
		}
		rewardFields[3][0] = -10;
		rewardFields[0][3] = -10;
		rewardFields[3][3] = 10;
		finalFields[3][0] = true;
		finalFields[0][3] = true;
		finalFields[3][3] = true;
		currentX = 0;
		currentY = 0;
	}
	
	public State reset() {
		currentX = 0;
		currentY = 0;
		isDone = false;
		return new State(new double[] {currentX, currentY});
	}
	
	public Observation step(int action) {
		if (isDone) {
			System.err.println("Can't make a move when the env is finished");
		}
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
		State state = new State(new double[] {currentX, currentY});
		double reward = rewardFields[currentX][currentY];
		boolean isFinal = finalFields[currentX][currentY];
		isDone = isFinal;
		return new Observation(state, reward, isFinal);
	}
}
