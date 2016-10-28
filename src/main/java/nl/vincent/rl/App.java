package nl.vincent.rl;

import nl.vincent.rl.algorithms.DQN;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
    	DQN dqn = new DQN();
    	dqn.run(10000, 200);
    }
}
