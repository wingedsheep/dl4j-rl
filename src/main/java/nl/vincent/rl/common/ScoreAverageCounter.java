package nl.vincent.rl.common;

public class ScoreAverageCounter {
	int number = 100;
	int scoresSeen = 0;
	int currentIndex = 0;
	double[] scores;
	
	public ScoreAverageCounter(int number) {
		this.number = number;
		this.scores = new double[number];
	}
	
	public void addScore(double score) {
		this.scores[currentIndex] = score;
		this.currentIndex ++;
		this.scoresSeen ++;
		if (currentIndex >= number) {
			currentIndex = 0;
		}
	}
	
	public double getAverage() {
		double sum = 0;
		double countTo = number;
		if (scoresSeen < number) {
			countTo = scoresSeen;
		}
		for (int i=0; i<countTo; i++) {
			sum += scores[i];
		}
		return sum / countTo;
	}
}
