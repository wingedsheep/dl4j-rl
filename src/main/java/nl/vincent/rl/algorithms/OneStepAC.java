package nl.vincent.rl.algorithms;

import nl.vincent.rl.networks.FullyConnected;

public class OneStepAC {
	FullyConnected policyModel;
	FullyConnected valueModel;
	double alpha = 0.01;
	double beta = 0.01;
	
	public OneStepAC() {
		policyModel = new FullyConnected(2, 4, new int[]{30, 30, 30}, 0.01);
		valueModel = new FullyConnected(2, 1, new int[]{30, 30, 30}, 0.01);
		
		for (int i=0; i < 100000 ; i++) {
			policyModel.train(new double[] {1, 2} , new double[] {1.5, 3.5, -3, 6} );
		}
		
		System.out.println(policyModel.predict(new double[] {1, 2} )[0]+","+policyModel.predict(new double[] {1, 2} )[1]+","+policyModel.predict(new double[] {1, 2} )[2]+","+policyModel.predict(new double[] {1, 2} )[3]);
	}

	public void run() {
		
	}
	
}
