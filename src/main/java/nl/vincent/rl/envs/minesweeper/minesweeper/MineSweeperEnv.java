package nl.vincent.rl.envs.minesweeper.minesweeper;

import nl.vincent.rl.common.Observation;
import nl.vincent.rl.common.State;
import nl.vincent.rl.envs.Environment;

public class MineSweeperEnv extends Environment {
	
	private MineSweeperGame mineSweeperGame;
	private int boardSize;
	private int toughness;
	
	public static int TOUGHNESS_EASY = 1;
	public static int TOUGHNESS_MODERATE = 2;
	public static int TOUGHNESS_HARD = 3;
	
	public static int FIELD_HIDDEN = -1;

	public MineSweeperEnv(int boardSize, int toughness) {
		super();
		this.boardSize = boardSize;
		this.toughness = toughness;
		this.actions = boardSize * boardSize;
		this.stateSize = boardSize * boardSize;
	}

	@Override
	public State reset() {
		if (mineSweeperGame != null && mineSweeperGame.isVisible()) {
			mineSweeperGame.setVisible(false); //you can't see me!
			mineSweeperGame.dispose(); //Destroy the JFrame object
		}
		mineSweeperGame = new MineSweeperGame(boardSize, toughness);
		mineSweeperGame.main(mineSweeperGame, boardSize);
		State state = getStateFromMineSweeperGame();
		return state;
	}

	@Override
	public Observation step(int action) {
		// size = 
		int actionY = action / boardSize;
		int actionX = action - (actionY * boardSize);
		double reward = mineSweeperGame.buttonClicked(actionX, actionY);
		State state = getStateFromMineSweeperGame();
		Observation obs = new Observation(state, reward, mineSweeperGame.isFinished());
		if (mineSweeperGame.isFinished()) {
			System.out.println("revealed: "+mineSweeperGame.getNoRevealed());
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			mineSweeperGame.setVisible(false); //you can't see me!
			mineSweeperGame.dispose(); //Destroy the JFrame object
		}
		return obs;
	}
	
	private State getStateFromMineSweeperGame() {
		int[][] observedField = mineSweeperGame.getObservation();
		double[] state = new double[boardSize * boardSize];
		int index = 0;
		for (int i = 0 ; i< observedField.length; i ++) {
    		for (int j = 0 ; j< observedField.length; j++) {
    			state[index] = observedField[i][j];
    			index ++;
        	}
    	}
		return new State(state);
	}

}
