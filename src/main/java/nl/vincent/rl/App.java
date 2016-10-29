package nl.vincent.rl;

import nl.vincent.rl.algorithms.DQN;
import nl.vincent.rl.envs.GridWorld;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
    	DQN dqn = new DQN.DQNBuilder()
    			.discountFactor(0.995)
    			.explorationRateDecay(0.995)
    			.hiddenLayers(new int[] {30, 30, 30})
    			.memorySize(100000)
    			.miniBatchSize(32)
    			.learningRate(0.01)
    			.startingExplorationRate(1.0).build();
    	
    	dqn.run(new GridWorld(), 10000, 200);
    }
}
