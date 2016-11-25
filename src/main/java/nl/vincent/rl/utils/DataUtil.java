package nl.vincent.rl.utils;

import java.util.ArrayList;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataUtil {
	
	public static INDArray toINDArray(double [] array) {
		INDArray vector = Nd4j.create(new int[] {1, array.length} );
        for (int i = 0; i < array.length; i++) {
        	vector.putScalar(i, array[i]);
        }
        return vector;
	}
	
	public static INDArray toINDArray(ArrayList<double[]> arrays) {
		INDArray matrix = Nd4j.create(new int[] {arrays.size(), arrays.get(0).length} );
        for (int i = 0; i < arrays.size(); i++) {
        	for (int j=0; j<arrays.get(i).length ; j ++) {
        		matrix.put(i, j, arrays.get(i)[j]);
        	}
        }
        return matrix;
	}
	
	public static INDArray toINDArray(double[][] arrays) {
		INDArray matrix = Nd4j.create(new int[] {arrays.length, arrays[0].length} );
        for (int i = 0; i < arrays.length; i++) {
        	for (int j=0; j<arrays[i].length ; j ++) {
        		matrix.put(i, j, arrays[i][j]);
        	}
        }
        return matrix;
	}
	
	public static double[] fromINDArrayVector(INDArray indArray) {
		int length = indArray.getRow(0).length();
        double[] result = new double[length];
		for (int i = 0; i < length; i++) {
			result[i] = indArray.getDouble(i);
		}
		return result;
	}
	
	public static double[][] fromINDArrayMatrix(INDArray indArray) {
		int rows = indArray.getColumn(0).length();
		int columns = indArray.getRow(0).length();
        double[][] result = new double[rows][columns];
        for (int i = 0; i < rows; i++) {
        	for (int j=0; j<columns ; j ++) {
        		result[i][j] = indArray.getDouble(i, j);
        	}
        }
		return result;
	}
}
