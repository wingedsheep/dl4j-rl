package nl.vincent.rl.common;

public abstract class Shape {
	private int rows = 0;
	private int cols = 0;
	
	public Shape(final int rows, final int cols) {
		this.rows = rows;
		this.cols = cols;
	}
	
	public int getRows() {
		return rows;
	}
	
	public int getCols() {
		return cols;
	}
}
