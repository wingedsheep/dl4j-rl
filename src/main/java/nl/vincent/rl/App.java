package nl.vincent.rl;

import nl.vincent.rl.algorithms.OneStepAC;
import nl.vincent.rl.envs.minesweeper.minesweeper.MineSweeperEnv;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
//    	DQN dqn = new DQN.DQNBuilder()
//    			.discountFactor(0.995)
//    			.explorationRateDecay(0.995)
//    			.hiddenLayers(new int[] {30, 30, 30})
//    			.memorySize(100000)
//    			.miniBatchSize(32)
//    			.learningRate(0.01)
//    			.startingExplorationRate(1.0).build();
//    	
//    	dqn.run(new GridWorld(), 10000, 200);
    	
//    	DQN dqn = new DQN.DQNBuilder()
//    			.discountFactor(0.995)
//    			.explorationRateDecay(0.995)
//    			.hiddenLayers(new int[] {100, 100, 100})
//    			.memorySize(100000)
//    			.miniBatchSize(32)
//    			.learningRate(0.001)
//    			.startingExplorationRate(1.0).build();
//    	
//    	dqn.run(new MineSweeperEnv(10, MineSweeperEnv.TOUGHNESS_EASY), 10000, 1000);
    	
 
    	
    	OneStepAC ac = new OneStepAC.Builder()
    			.discountFactor(0.999)
    			.alpha(1e-3)
    			.beta(1e-3)
    			.hiddenLayersPolicy(new int[] {30, 30, 30})
    			.hiddenLayersValue(new int[] {30, 30, 30})
    			.build();

//    	ac.run(new GridWorld(), 10000, 200);
    	ac.run(new MineSweeperEnv(10, MineSweeperEnv.TOUGHNESS_EASY), 100000, 1000);
    }
}
