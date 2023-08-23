package elki;

import java.util.Arrays;
import java.util.Random;
import elki.math.linearalgebra.VMath;

public class moVMF {

    private int k;
    private int d;
    private int maxIterations;

    /**
     * Constructor
     * @param k number of components
     * @param d dimension
     * @param maxIterations maximum number of iterations
     */

    public moVMF(int k, int d, int maxIterations){

        this.k = k;
        this.d = d;
        this.maxIterations = maxIterations;

    }

    /**
     * A method to train the model
     * @param numClusters number of clusters
     * @param fWeights given  alphas
     * @param check check if the initialisation is complete
     * @param input the data input
     * @param init
     * @param ranState random State
     * @param posteriorT hard or soft movmf
     * @param maxIter maximum number of iterations
     * @return
     */

    //method too long, will split it for more efficiency

    public static double[][] train (int numClusters, double[] fWeights,
                                    boolean check, double[][] input, String init, long ranState,
                                    String posteriorT, double tolerance, int maxIter) {

        //start by initialising the centers using a helping method
        int nExamples = input.length;
        int nFeatures = input[0].length;
        double[][] centers = initUnitCenters(input, numClusters, init, ranState); //helping method

        //initialise the probabilities alpha
        double[] sWeights;
        if (fWeights == null) {
            sWeights = new double[numClusters];
            Arrays.fill(sWeights, 1.0 / numClusters);
        } else {
            sWeights = fWeights;
        }

        // initialise kappas
        double[] kappas = new double[numClusters];
        Arrays.fill(kappas, 1.0);

        if (check) {
            System.out.println("Initialization complete");
        }


        double[][] posterior = new double[0][];
        double[] newCenters = new double[0];
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] centersPrev = centers.clone();

            // Expectation step
            posterior = expectation(input, centers, sWeights, kappas, posteriorT);

            // Maximization step
            double[][] result = maximization(input, posterior, fWeights);
            newCenters = result[0];
            sWeights = result[1];
            kappas = result[2];

            // Check convergence
            double tolcheck = squaredNorm(centersPrev, centers);
            if (tolcheck <= tolerance) {
                if (check) {
                    System.out.printf("Converged at iteration %d: center shift %e within tolerance %e%n", iter, tolcheck, tolerance);
                }
                break;
            }
        }

        // Compute labels
        double[] labels = new double[nExamples];
        for (int ee = 0; ee < nExamples; ee++) {
            //labels[ee] = VMath.argmax(posterior, ee, 0)
        }

        // Compute inertia
        double inertia = inertiaFromLabels(input, centers, labels);

        return new double[][]{newCenters, labels, new double[]{inertia}, sWeights, kappas, posterior};

    }

    /**
     * helping method to initialise the centers depending on the variable unit
     * @param arr array
     * @param nClusters number of clusters
     * @param init defines the initialisation type
     * @param ranState the random state
     * @return
     */

    private static double[][] initUnitCenters(double[][] arr, int nClusters, String init, long ranState){
        Random random = new Random(ranState);
        int nExamples = arr.length;
        int features = arr[0].length;

        double[][] centers;


        switch (init) {
            case "spherical-k-means":
                // TODO: Implement spherical-k-means initialization
                throw new UnsupportedOperationException("Spherical k-means initialization is not implemented yet");

            case "random":
                centers = new double[nClusters][features];
                for (int cc = 0; cc < nClusters; cc++) {
                    randomUnitNormVector(random, centers[cc]);
                }
                break;

            case "k-means++":
                // TODO: Implement k-means++ initialization
                throw new UnsupportedOperationException("K-means++ initialization is not implemented yet");

            case "random-orthonormal":
                centers = new double[nClusters][features];
                double[][] randomData = new double[features][features];
                //TODO randomOrthonormalVectors(random, randomData);
                for (int cc = 0; cc < nClusters; cc++) {
                    System.arraycopy(randomData[cc], 0, centers[cc], 0, features);
                }
                break;

            case "random-class":
                centers = new double[nClusters][features];
                for (int cc = 0; cc < nClusters; cc++) {
                    while (squaredNorm(centers[cc]) == 0.0) {
                        int[] labels = new int[nExamples];
                        random.ints(0, nClusters).limit(nExamples).toArray(labels);

                        for (int ee = 0; ee < nExamples; ee++) {
                            addVectorsInPlace(centers[cc], arr[ee], labels[ee] == cc ? 1.0 : 0.0);
                        }
                    }
                }
                break;

            default:
                throw new IllegalArgumentException("Invalid init value: " + init);
        }

        return centers;
    }


    /**
     *
     * @param arr
     * @param centers
     * @param weights
     * @param concentrations
     * @param posterior
     * @return
     */
    private static double[][] expectation(double[][] arr, double[][] centers, double[] weights,
                                          double[] concentrations, String posterior){
        int nExamples = arr.length;
        int clusters = centers.length; // centers determine how many clusters there are
        double[][] posteriorT = new double[nExamples][clusters];

        for(int ee = 0; ee < nExamples; ee++){
            double[] x = arr[ee];
            double[] logprob = new double[clusters];

            for(int cc = 0; cc < clusters; cc++ ){
                logprob[cc] = Math.log(weights[cc]) + concentrationLogpdf(x, centers[cc], concentrations[cc]);

            }
            double maxLogprob = Arrays.stream(logprob).max().orElse(0.0);
            double sumLogprob = 0.0;

            for(int cc = 0; cc < clusters; cc++){
                double weight = Math.exp(logprob[cc]-maxLogprob);
                posteriorT[ee][cc] = weight;
                sumLogprob += weight;
            }

            double norm = 1.0 / sumLogprob;
            for(int cc = 0; cc < clusters; cc++){
                posteriorT[ee][cc] *=norm;
            }
        }
        return posteriorT;
    }

    /**
     *
     * @param arr
     * @param posterior
     * @param forceWeights
     * @return
     */
    private static double[][] maximization(double[][] arr, double[][] posterior, double [] forceWeights){
        int nExamples = arr.length;
        int features = arr[0].length;
        int clusters = posterior[0].length; // The posterior matrix from the expectation step.
        double[][] centers = new double[clusters][features];
        double[] weights = new double[clusters];
        double[] concentrations = new double[clusters];

        for (int cc = 0; cc < clusters; cc++) {
            double weightsSum = 0.0;
            double[] weightedSum = new double[features];
            double[] weightedDotSum = new double[features];
            double concentrationSum = 0.0;

            for(int ee = 0; ee <nExamples; ee++){
                double weight = posterior[ee][cc];
                weightsSum += weight;
                addVectorsInPlace(weightedSum, arr[ee], weight);
                addVectorsInPlace(weightedDotSum, multiplyVectors(arr[ee], arr[ee]), weight);
                concentrationSum += weight * dotProduct(arr[ee], arr[ee]);
            }
            weights[cc] = forceWeights != null ? forceWeights[cc] : weightsSum / nExamples;
            concentrations[cc] = concentrationSum / weightsSum;

            if (weightsSum != 0.0) {
                multiplyVectorInPlace(weightedSum, 1.0 / weightsSum);
            }
            if (concentrationSum != 0.0) {
                concentrations[cc] /= (features * weightsSum);
            }
            System.arraycopy(weightedSum, 0, centers[cc], 0, features);
        }

        return new double[][] {centers, weights, concentrations };
    }

    /**
     *
     * @param x the first vector
     * @param y the second vector
     * @param fac the factor
     */
    private static void addVectorsInPlace(double[] x, double[] y, double fac){
        for(int i = 0; i < x.length; i++){
            x[i] += fac * y[i];
        }
    }

    /**
     *
     * @param x
     * @param center
     * @param concentration
     * @return
     */
    private static double concentrationLogpdf(double[] x, double[] center, double concentration){
        //TODO implement the method as to calculate the logpdf
        return 0.0;
    }

    /**
     * private method to compute the dot product
     * @param x first value
     * @param y second value
     * @return dot product of the two values
     */

    private static double dotProduct( double[] x, double[] y){
        double dot = 0.0;
        for(int i = 0; i < x.length; i++){
            dot += x[i] * y[i];
        }
        return dot;
    }

    // argmax already exists in elki (to be used from there)

    /**
     *
     * @param x
     * @param y
     * @return
     */
    private static double[] multiplyVectors(double[] x, double[] y){
        int n = x.length;
        double[] result = new double[n];
        for (int i = 0; i < n; i++) {
            result[i] = x[i] * y[i];
        }
        return result;
    }

    /**
     *
     * @param vector
     * @param scalar
     */
    private static void multiplyVectorInPlace(double[] vector, double scalar) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }
    }

    /**
     *
     * @param random
     * @param vector
     */
    private static void randomUnitNormVector(Random random, double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            vector[i] = random.nextGaussian();
        }
        normalizeVector(vector);
    }

    /**
     * method that normalizes a vector
     * @param vector the vector to be normalized
     */

    private static void normalizeVector(double[]  vector){
        double norm = 0.0;
        for (double value : vector) {
            norm += value * value;
        }
        norm = Math.sqrt(norm);
        if (norm != 0.0) {
            for (int i = 0; i < vector.length; i++) {
                vector[i] /= norm;
            }
        }
    }

    private static double squaredNorm(double[][] x, double[][] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += squaredNorm(x[i], y[i]);
        }
        return sum;
    }

    private static double squaredNorm(double[] x, double[] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double diff = x[i] - y[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     *
     * @param X
     * @param centers
     * @param labels
     * @return
     */
    private static double inertiaFromLabels(double[][] X, double[][] centers, double[] labels) {
        double inertia = 0.0;
        for (int ee = 0; ee < X.length; ee++) {
            double[] x = X[ee];
            int label = (int) labels[ee];
            double[] center = centers[label];
            inertia += squaredNorm(x, center);
        }
        return inertia;
    }
}

