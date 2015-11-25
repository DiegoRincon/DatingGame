import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Random;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.jblas.DoubleMatrix;
import org.jblas.Solve;
//import org.ejml.simple.SimpleMatrix;


public class Matchmaker {
    public Map<Integer, Boolean> mapOfWeightAndSign;
    public Map<Candidate, Double> mapCandidateToResult;
    public double[] positiveWeights;
    public double[] negativeWeights;
    private int numOfAttributes;
    public Candidate previousCandidate;
    private Random random;
    private double learningRate = -100;
    private boolean isIdeal = true;
    private boolean firstTurn = true;
    private double pastScore;
    private List<Integer> goodIndexes;
    private Integer[] pastIndexes;
    private Map<Integer, Double> mapOfGoodIndexes;
    private Map<Integer, Double> mapOfPastModIndexes;
    private double[] bestWeights = null;
    private double[] newWeights;
    private int lastIndex = 0;
    private Map<Integer, Integer> mapOfBadIndexes;
    private static final int MIN_RANGE = -1;
    private static final int MAX_RANGE = 1;
    private static double EPSILON = 1.0/2000000.0;
    private static final double THRESHOLD = 450000;
    private static double VARIANCE = 0.05;
    private static int REMOVE_INDEX = Math.min(2, (int)(VARIANCE*20));
    private int turnNum = 0;
    
    
    public Matchmaker(int numOfAttributes) {
//	EPSILON = 1.0/epsilon;
//	VARIANCE = variance;
	this.numOfAttributes = numOfAttributes;
	this.mapCandidateToResult = new HashMap<Candidate, Double>();
	this.mapOfWeightAndSign = new HashMap<Integer, Boolean>();
	this.random = new Random();
	this.goodIndexes = new ArrayList<Integer>();
	this.mapOfBadIndexes = new HashMap<Integer, Integer>();
	this.mapOfGoodIndexes = new HashMap<Integer, Double>();
	this.mapOfPastModIndexes = new HashMap<Integer, Double>();
//	System.out.println(EPSILON);
    }
    
    public Candidate driver() {
	this.turnNum++;
	double[] guessWeights = calculateWeights();
	this.newWeights = guessWeights;
	Candidate candidate = (this.isIdeal || this.turnNum == 20)? generateIdealCandidate(guessWeights): generateTestCandidate(guessWeights);
	this.firstTurn = false;
	this.isIdeal = !this.isIdeal;
	return candidate;
    }
    
    public void processCandidate(Candidate candidate, double score) {
	this.mapCandidateToResult.put(candidate, score);
	if (score < this.pastScore) {
	    updateDictOfBadIndexes(this.mapOfBadIndexes, this.pastIndexes);
	} else {
	    updateDictOfGoodIndexes(this.mapOfGoodIndexes, this.mapOfPastModIndexes);
	    this.bestWeights = this.newWeights;
	}
	this.pastScore = score;
    }
       
    private void updateDictOfGoodIndexes(Map<Integer, Double> map, Map<Integer, Double> pastMap) {
	for (Entry<Integer, Double> entry : pastMap.entrySet()) {
	    if (!map.containsKey(entry.getKey())) {
		map.put(entry.getKey(), entry.getValue());
	    }
	}
    }
    
    private void updateDictOfBadIndexes(Map<Integer, Integer> map, Integer[] array) {
	if (array == null)
	    return;
	for (Integer i : array) {
	    if (map.containsKey(i)) {
		map.put(i, map.get(i)+1);
	    } else {
		map.put(i, 1);
	    }
	}
    }
    
    private Candidate generateIdealCandidate(double[] weights) {
	System.out.println("Ideal");
//	System.out.println(Arrays.toString(weights));
	double[] attributes = new double[weights.length];
	for (int i=0; i<weights.length; i++) {
	    if (weights[i] > 0) {
		attributes[i] = 1;
	    } else {
		attributes[i] = 0;
	    }
	}
	Candidate candidate = new Candidate(this.numOfAttributes);
	candidate.attributes = attributes;
	this.previousCandidate = candidate;
	this.pastIndexes = new Integer[(int)(this.numOfAttributes*VARIANCE)];
	return candidate;
    }
    
    private Candidate generateProbingCandidate(double[] weights) {
	double[] pastAttributes = this.previousCandidate.attributes;
	double[] newAttributes = new double[weights.length];
	for (int i=0; i < newAttributes.length; i++) {
	    double diffPosAndNeg = Math.abs(this.positiveWeights[i] - this.negativeWeights[i]);
//	    System.out.println("Difference " + diffPosAndNeg);
	    if (diffPosAndNeg > THRESHOLD) {
		newAttributes[i] = pastAttributes[i];
	    } else {
		newAttributes[i] = (pastAttributes[i] == 0)? 1: 0;
	    }
	}
	Candidate candidate = new Candidate(this.numOfAttributes);
	candidate.attributes = newAttributes;
	return candidate;
    }
    
    private Candidate generateTestCandidate(double[] weights) {
	System.out.println("Test");
	int numOfChanges = (int)(this.numOfAttributes*VARIANCE);
	double[] newAttributes = new double[weights.length];
	if (!this.firstTurn) {
	    double[] pastAttributes = this.previousCandidate.attributes;
	    System.arraycopy(pastAttributes, 0, newAttributes, 0, pastAttributes.length);
	}
	for (Entry<Integer, Double> entry : this.mapOfGoodIndexes.entrySet()) {
	    newAttributes[entry.getKey()] = entry.getValue();
	}

	this.pastIndexes = new Integer[numOfChanges];
	for (int i = 0; i < numOfChanges; i++) {
	    int index = random.nextInt(this.numOfAttributes);
	    while (this.mapOfBadIndexes.containsKey(index) && this.mapOfBadIndexes.get(index) > REMOVE_INDEX) {
		index = random.nextInt(this.numOfAttributes);
	    }

	    if (Math.abs(weights[index]) >= 0.02) {
		i--;
		continue;
	    }
	    
	    this.pastIndexes[i] = index;
	    double newValue = (newAttributes[index] == 0)? 1: 0;;
	    newAttributes[index] = newValue;
	    this.mapOfPastModIndexes.put(index, newValue);
	}
	Candidate candidate = new Candidate(this.numOfAttributes);
	candidate.attributes = newAttributes;
	return candidate;
    }
    
    private Candidate generateTestConsecutiveCandidate(double[] weights) {
	System.out.println("Test");
	double[] pastAttributes = this.previousCandidate.attributes;
	double[] newAttributes = new double[weights.length];
	int numOfChanges = (int)(this.numOfAttributes*VARIANCE);
	System.arraycopy(pastAttributes, 0, newAttributes, 0, pastAttributes.length);
	this.pastIndexes = new Integer[numOfChanges];
	for (int i = 0; i < numOfChanges; i++) {
	    if (Math.abs(weights[this.lastIndex]) > 0.01 || this.goodIndexes.contains(this.lastIndex)) {
		i--;
		this.lastIndex = (this.lastIndex + 1 < this.numOfAttributes)? this.lastIndex + 1: 0;		
		continue;
	    }
	    this.pastIndexes[i] = this.lastIndex;
	    newAttributes[this.lastIndex] = (newAttributes[this.lastIndex] == 0)? 1: 0;
	    this.lastIndex = (this.lastIndex + 1 < this.numOfAttributes)? this.lastIndex + 1: 0;
	}
	Candidate candidate = new Candidate(this.numOfAttributes);
	candidate.attributes = newAttributes;
	return candidate;	
    }

    public double meanSquaredError(double[] guessWeights, Map<Candidate, Double> map) {
	double sum = 0;
	int numOfCandidates = map.size();
	for (Entry<Candidate, Double> entry : map.entrySet()) {
	    double guessValue = dotProduct(guessWeights, entry.getKey().attributes);
	    sum += Math.pow(guessValue - entry.getValue(), 2);
	}
	return sum/numOfCandidates;
    }
    
    public double cost(double[] guessWeights, Map<Candidate, Double> map) {
//	double alpha = getCoefficient(map, alpha, alpha, alpha);
	return meanSquaredError(guessWeights, map);
    }
    
    public double[] getRandomVector(int n) {
	double[] vector = new double[n];
	Random random = new Random();
	for (int i=0; i<n; i++) {
	    vector[i] = MIN_RANGE + (MAX_RANGE - MIN_RANGE) * random.nextDouble();
	}
	return vector;
    }
    
    public double[] calculateWeights() {
	try {
	    if (this.learningRate == -100) {
		this.learningRate = getCoefficient(this.mapCandidateToResult, 0.00001, 0.1, 1.5);
		System.out.println("Done learning rate");
	    } else if (this.turnNum == 10) {
		this.learningRate = getCoefficient(this.mapCandidateToResult, 0.00001, 0.1, 5);
		System.out.println("Done learning rate");		
	    }
	    double[] weights = gradientDescent(this.mapCandidateToResult, this.learningRate, false);
	    int numOfZero = 0;
	    for (double d : weights) {
		if (d == 0)
		    numOfZero++;
	    }
//	    System.out.println(numOfZero);
	    return weights;
	} catch (InterruptedException e) {
	    e.printStackTrace();
	} catch (ExecutionException e) {
	    e.printStackTrace();
	}
	return null;
    }
    
    public double magnitudeOfVector(double[] vector) {
	double sum = 0;
	for (double d : vector) {
	    sum += d;
	}
	return sum;
    }
    
    public double diffOfVectors(double[] vector, double[] vector2) {
	double sum = 0.0;
	for (int i=0; i<vector.length; i++) {
	    sum += Math.abs(vector[i] - vector2[i]);
	}
	return sum;
    }
    
    public double[] gradientDescent(Map<Candidate, Double> map, double learningRate, boolean isTrain) {
	if (!isTrain) {
	    this.positiveWeights = new double[this.numOfAttributes];
	    this.negativeWeights = new double[this.numOfAttributes];
	}
//	double[] weights = Person.createRandomWeights(this.numOfAttributes);
	double[] weights = (this.bestWeights == null)? new double[this.numOfAttributes] : this.bestWeights;
	double difference = 1;
	double[] past = null;
//	System.out.println("Gradient Descent started");
	while (difference > EPSILON) {
	    double[] gradientVector = new double[this.numOfAttributes];
	    for (Entry<Candidate, Double> entry : map.entrySet()) {
		Candidate candidate = entry.getKey();
		Double value = entry.getValue();
		double dotProd = dotProduct(candidate.attributes, weights);
		double diff = dotProd - value;
		//Update gradient index
		for (int j=0; j<this.numOfAttributes; j++) {
		    gradientVector[j] = gradientVector[j] + (diff*candidate.attributes[j]);
		}
	    }
	    //Update weights
	    for (int j=0; j<this.numOfAttributes; j++) {
		weights[j] = weights[j] - learningRate*gradientVector[j];
	    }
	    if (!isTrain) {
		//Update positive and negative;
		for (int j=0; j<this.numOfAttributes; j++) {
		    if (weights[j] < 0) {
			this.negativeWeights[j]++;
		    } else {
			this.positiveWeights[j]++;
		    }
		}
	    }
	    if (past == null) {
		past = new double[weights.length];
	    } else {
		difference = diffOfVectors(weights, past);
	    }
	    System.arraycopy(weights, 0, past, 0, weights.length);
	}
	for (int i=0; i<weights.length; i++) {
	    weights[i] = getDoubleWith2Decimals(weights[i]);
	}
//	System.out.println("Count epsilon: " + count);
//	System.out.println("Validated: " + Person.validateWeights(weights));
	return weights;
    }
    
    public static double getDoubleWith2Decimals(double d) {
	return Math.round(d*100)/100.0;
    }
    
    public double getCoefficient(Map<Candidate, Double> map, double start, double end, double increment) throws InterruptedException, ExecutionException {
	double bestCost = Double.MAX_VALUE;
	double bestEta = 0;
	ExecutorService executor = Executors.newFixedThreadPool(8);
	List<Future<Tuple<Candidate, double[]>>> tasks = new ArrayList<Future<Tuple<Candidate, double[]>>>(); 
	for (double learningRate=start; learningRate<=end; learningRate*=increment) {
	    for (Entry<Candidate, Double> entry : map.entrySet()) {
		Candidate candidate = entry.getKey();
		Map<Candidate, Double> mapWithoutCandidate = getMapWithoutCandidate(map, candidate);
		TrainCallable gdc = new TrainCallable(mapWithoutCandidate, learningRate, candidate);
		tasks.add(executor.submit(gdc));
	    }
	    double currCost = 0;
	    for (Future<Tuple<Candidate, double[]>> future : tasks) {
		Tuple<Candidate, double[]> tuple = future.get();
		Candidate candidate = tuple.first;
		double[] weights = tuple.second;
		if (weights == null) {
		    continue;
		}
		Map<Candidate, Double> singlesMap = new HashMap<Candidate, Double>();
		singlesMap.put(candidate, map.get(candidate));
		currCost = currCost + cost(weights, singlesMap);
	    }
	    if (currCost < bestCost) {
		bestCost = currCost;
		bestEta = learningRate;
	    }
	}
	executor.shutdown();
	return bestEta;
    }
    
    private class TrainCallable implements Callable<Tuple<Candidate, double[]>> {
	private Map<Candidate, Double> map;
	private double learningRate;
	private Candidate candidate;
	public TrainCallable(Map<Candidate, Double> map, double learningRate, Candidate candidate) {
	    this.map = map;
	    this.learningRate = learningRate;
	    this.candidate = candidate;
	}
	public Tuple<Candidate, double[]> call() throws Exception {
	    return new Tuple<Candidate, double[]>(this.candidate, gradientDescent(this.map, this.learningRate, true));
	}
	
    }
    
    private Map<Candidate, Double> getMapWithoutCandidate(Map<Candidate, Double> map, Candidate candidate) {
	Map<Candidate, Double> newMap = new HashMap<Candidate, Double>(map);	
	if (newMap.remove(candidate) == null) {
	    throw new RuntimeException("No such candidate!!");
	}
	return newMap;
    }

    public double[] train(Map<Candidate, Double> map) {
	double[][] candidates = new double[map.size()][this.numOfAttributes];
	double[][] scores = new double[map.size()][1];
	double[] weights = new double[this.numOfAttributes];
	int index=0;
	for (Entry<Candidate, Double> entry : map.entrySet()) {
	    Candidate candidate = entry.getKey();
	    candidates[index] = candidate.attributes;
	    scores[index][0] = entry.getValue();
	    index++;
	}
	DoubleMatrix candidatesMatrix = new DoubleMatrix(candidates);
	DoubleMatrix scoresMatrix = new DoubleMatrix(scores);
	DoubleMatrix tmpMatrix = candidatesMatrix.transpose().mmul(candidatesMatrix);
	DoubleMatrix tmp = Solve.pinv(tmpMatrix);
	DoubleMatrix tmp2 = tmp.mmul(candidatesMatrix.transpose());
	DoubleMatrix weightsMatrix = tmp2.mmul(scoresMatrix);
	weights = weightsMatrix.data;
	return weights;
    }

    public static double dotProduct(double[] vector1, double[] vector2) {
	if (vector1.length != vector2.length) {
	    return -1;
	}
	double score = 0;
	for (int i=0; i<vector1.length; i++) {
	    score += vector1[i] * vector2[i];
	}
	return score;
    }

}
