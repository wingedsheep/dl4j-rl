package nl.vincent.rl;

import java.io.IOException;

import nl.vincent.rl.algorithms.DQN;
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
    	
    	DQN dqn = new DQN.DQNBuilder()
    			.discountFactor(0.995)
    			.explorationRateDecay(0.995)
    			.hiddenLayers(new int[] {250, 250, 250, 250, 250})
    			.memorySize(1000000)
    			.miniBatchSize(64)
    			.learningRate(0.0001)
    			.environment(new MineSweeperEnv(10, MineSweeperEnv.TOUGHNESS_EASY))
    			.startingExplorationRate(1.0).build();
    	
    	try {
			dqn.loadModel1("qModel1_9000.zip");
			dqn.loadModel2("qModel2_9000.zip");
    	} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	
    	dqn.run(100000, 1000);
    	
 
    	
//    	OneStepAC ac = new OneStepAC.Builder()
//    			.discountFactor(0.999)
//    			.alpha(1e-3)
//    			.beta(1e-3)
//    			.hiddenLayersPolicy(new int[] {30, 30, 30})
//    			.hiddenLayersValue(new int[] {30, 30, 30})
//    			.build();

//    	ac.run(new GridWorld(), 10000, 200);
//    	ac.run(new MineSweeperEnv(10, MineSweeperEnv.TOUGHNESS_EASY), 100000, 1000);
    }
}
