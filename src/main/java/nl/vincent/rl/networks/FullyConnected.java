package nl.vincent.rl.networks;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class FullyConnected {
	private MultiLayerNetwork model;
	private int inputs, outputs;
	
	public FullyConnected(int inputs, int outputs, int[] hiddenLayers, double learningRate) {
		this.inputs = inputs;
		this.outputs = outputs;
		MultiLayerConfiguration conf = null;
		if (hiddenLayers.length == 0) {
	        conf = new NeuralNetConfiguration.Builder()
	                .iterations(1)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .learningRate(learningRate)
	                .updater(Updater.RMSPROP)
	                .list()
	                .layer(0, new OutputLayer.Builder(LossFunction.SQUARED_LOSS)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("softmax").weightInit(WeightInit.XAVIER)
	                        .nIn(inputs).nOut(outputs).build())
	                .pretrain(false).backprop(true).build();
		} else {
			NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
	                .iterations(1)
	                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
	                .learningRate(learningRate)
	                .updater(Updater.RMSPROP)
	                .list()
	                .layer(0, new DenseLayer.Builder().nIn(inputs).nOut(hiddenLayers[0])
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("relu")
	                        .build());
	        for (int i = 0 ; i < hiddenLayers.length - 1 ; i++) {
	        	builder.layer(i+1, new DenseLayer.Builder().nIn(hiddenLayers[i]).nOut(hiddenLayers[i+1])
                        .weightInit(WeightInit.XAVIER)
                        .activation("relu")
                        .build());
	        }
	        conf =  builder.layer(hiddenLayers.length, new OutputLayer.Builder(LossFunction.SQUARED_LOSS)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("identity").weightInit(WeightInit.XAVIER)
	                        .nIn(hiddenLayers[hiddenLayers.length - 1]).nOut(outputs).build())
	                .pretrain(false).backprop(true).build();
		}
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        this.model = model;
	}
	
	/**
	 * Train the network on an array of input size with a state
	 * @param input
	 * @return
	 */
	public double[] predict(double[] input) {
        INDArray inputVector = Nd4j.create(new int[] {1, inputs} );
        for (int i = 0; i < inputs; i++) {
        	inputVector.putScalar(i, input[i]);
        }
        INDArray resultVector = model.output(inputVector);
        double[] result = new double[outputs];
		for (int i = 0; i < outputs; i++) {
			result[i] = resultVector.getDouble(i);
		}
		return result;
	}
	
	public void train(double[] input, double[] output) {
		INDArray inputVector = null;
		INDArray outputVector = null;
		inputVector = Nd4j.create(new int[] {1, inputs} );
        for (int i = 0; i < inputs; i++) {
        	inputVector.putScalar(i, input[i]);
        }
        outputVector = Nd4j.create(new int[] {1, outputs} );
        for (int i = 0; i < outputs; i++) {
        	outputVector.putScalar(i, output[i]);
        }
		model.fit(inputVector, outputVector);
		System.out.println(model.score());
	}
	
	public void train(ArrayList<double[]> input, ArrayList<double[]> output) {
		INDArray inputVector = null;
		INDArray outputVector = null;
		inputVector = Nd4j.create(new int[] {input.size(), inputs} );
        for (int i = 0; i < input.size(); i++) {
        	for (int j=0; j<input.get(i).length ; j ++) {
        		inputVector.put(i, j, input.get(i)[j]);
        	}
        }
        outputVector = Nd4j.create(new int[] {output.size(), outputs} );
        for (int i = 0; i < output.size(); i++) {
        	for (int j=0; j<output.get(i).length ; j ++) {
        		outputVector.put(i, j, output.get(i)[j]);
        	}
        }
		model.fit(inputVector, outputVector);
	}
	
	public void saveModel(String path) throws IOException {
		//Save the model
	    File locationToSave = new File(path);      //Where to save the network. Note: the file is in .zip format - can be opened externally
	    boolean saveUpdater = true;                                     //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
	    ModelSerializer.writeModel(this.model, locationToSave, saveUpdater);
	}
}
