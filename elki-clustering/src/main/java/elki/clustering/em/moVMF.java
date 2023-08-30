package elki.clustering.em;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.clustering.kmeans.initialization.RandomlyChosen;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.MeanModel;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.database.relation.Relation;
import elki.distance.CosineDistance;
import elki.distance.NumberVectorDistance;
import elki.math.linearalgebra.VMath;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.Flag;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import org.apache.commons.math3.special.BesselJ;
import java.util.Arrays;
import java.util.Random;


public class moVMF<V extends NumberVector, M extends Model> implements ClusteringAlgorithm<Clustering<M>>{

    /**
   * Class to choose the initial means
   */
    protected KMeansInitialization initializer;

    protected NumberVectorDistance<? super V> distance = CosineDistance.STATIC;

    private int k;
    private int maxIterations;
    private int minIter;
    //private int dimension;

    /**
     * Constructor
     * @param k number of components
     * @param maxIterations maximum number of iterations
     * @param minIter minimum number of Iterations
     * @param soft decides if it's a soft or hard clustering
     */

    public moVMF(int k, int minIter, int maxIterations, double delta, boolean soft, KMeansInitialization initializer){
        this.k = k;
        this.maxIterations = maxIterations;
        this.minIter = minIter;
        this.initializer = initializer;
    }

    /**
     * A method to train the model
     * @param numClusters number of clusters
     * @param fWeights given  alphas
     * @param check check if the initialisation is complete
     * @param input the data input
     * @param init
     * @param ranState random State
     * @param maxIter maximum number of iterations
     * @return
     */

    //method too long, will split it for more efficiency

    public static double[][] train (int numClusters, double[] fWeights,
                                    boolean check, NumberVector[] input, String init, long ranState,
                                    double tolerance, int maxIter) {

        //start by initialising the centers using a helping method
        int nExamples = input.length;
        int nFeatures = input[0].getDimensionality();
        NumberVector[] centers = initUnitCenters(input, numClusters, init, ranState); //helping method

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
            NumberVector[] centersPrev = centers.clone();

            // Expectation step
            posterior = expectation(input, centers, sWeights, kappas);

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
            labels[ee] = VMath.argmax(posterior, ee, nExamples);
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

    private static NumberVector[] initUnitCenters(NumberVector[] arr, int nClusters, String init, long ranState){
        Random random = new Random(ranState);
        int nExamples = arr.length;
        int features = arr[0].getDimensionality();

        NumberVector[] centers = new NumberVector[nClusters];

        switch (init) {
            case "spherical-k-means":
                // TODO: Implement spherical-k-means initialization
                // exists in elki
                throw new UnsupportedOperationException("Spherical k-means initialization is not implemented yet");

            case "random":
                for (int cc = 0; cc < nClusters; cc++) {
                    randomUnitNormVector(random, centers[cc].toArray());
                }
                break;

            case "k-means++":
                // TODO: Implement k-means++ initialization
                throw new UnsupportedOperationException("K-means++ initialization is not implemented yet");

            case "random-class":
                for (int cc = 0; cc < nClusters; cc++) {
                    while (squaredNorm(centers[cc].toArray()) == 0.0) {
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
     * @return
     */
    private static double[][] expectation(NumberVector[] arr, NumberVector[] centers, double[] weights,
                                          double[] concentrations){
        int nExamples = arr.length; // the input
        int clusters = centers.length; // centers determine how many clusters there are
        double[][] posterior = new double[nExamples][clusters];

        for(int ee = 0; ee < nExamples; ee++){
            NumberVector x = arr[ee];
            double[] logprob = new double[clusters];

            for(int cc = 0; cc < clusters; cc++ ){
                logprob[cc] = Math.log(weights[cc]) + vonMisesFisherLogPDF(x, centers[cc], concentrations[cc], x.getDimensionality());

            }
            double maxLogprob = Arrays.stream(logprob).max().orElse(0.0);
            double sumLogprob = 0.0;

            for(int cc = 0; cc < clusters; cc++){
                double weight = Math.exp(logprob[cc]-maxLogprob);
                posterior[ee][cc] = weight;
                sumLogprob += weight;
            }

            double norm = 1.0 / sumLogprob;
            for(int cc = 0; cc < clusters; cc++){
                posterior[ee][cc] *=norm;
            }
        }
        return posterior;
    }

    /**
     *
     * @param arr
     * @param posterior
     * @param forceWeights
     * @return
     */
    private static double[][] maximization(NumberVector[] arr, double[][] posterior, double [] forceWeights){
        int nExamples = arr.length;
        int features = arr[0].getDimensionality();
        int clusters = posterior[0].length; // The posterior matrix from the expectation step.
        NumberVector[] centers = new NumberVector[clusters];
        double[] weights = new double[clusters];
        double[] concentrations = new double[clusters];

        for (int cc = 0; cc < clusters; cc++) {
            double weightsSum = 0.0;
            double[] weightedSum = new double[arr[0].getDimensionality()];
            double[] weightedDotSum = new double[arr[0].getDimensionality()];
            double concentrationSum = 0.0;

            for(int ee = 0; ee <nExamples; ee++){
                double weight = posterior[ee][cc];
                weightsSum += weight;
                addVectorsInPlace(toNumberVector(weightedSum), arr[ee], weight);
                addVectorsInPlace(toNumberVector(weightedDotSum), multiplyVectors(arr[ee], arr[ee]), weight);
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

        return new NumberVector[] {centers, toNumberVector(weights), toNumberVector(concentrations)};
    }


    public static NumberVector toNumberVector(double[] array){
        return new DoubleVector(array);
    }

    /**
     *
     * @param x the first vector
     * @param y the second vector
     * @param fac the factor
     */
    private static void addVectorsInPlace(NumberVector x, NumberVector y, double fac){
        double result = 0.0;
        for(int i = 0; i < x.getDimensionality(); i++){
            result += x.doubleValue(i) + (fac * y.doubleValue(i));
        }
    }

    /**
     * private method to compute the dot product
     * @param x first value
     * @param y second value
     * @return dot product of the two values
     */

    public static double dotProduct(NumberVector x, NumberVector y) {
        double result = 0.0;
        for (int i = 0; i < x.getDimensionality(); i++) {
            result += x.doubleValue(i) * y.doubleValue(i);
        }
        return result;
    }
    // argmax already exists in elki (to be used from there)

    /**
     *
     * @param x
     * @param y
     * @return
     */
    public static NumberVector multiplyVectors(NumberVector x, NumberVector y) {
        int dimensionality = x.getDimensionality();
        double[] resultArray = new double[dimensionality];

        for (int i = 0; i < dimensionality; i++) {
            resultArray[i] = x.doubleValue(i) * y.doubleValue(i);
        }
        return new DoubleVector(resultArray);
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

    // ob alle 3 gebraucht werden?
    private static double squaredNorm(NumberVector[] x, NumberVector[] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += squaredNorm(x[i], y[i]);
        }
        return sum;
    }

    private static double squaredNorm(NumberVector x, NumberVector y) {
        double sum = 0.0;
        int dimensionality = x.getDimensionality();
        for (int i = 0; i < dimensionality; i++) {
            double diff = x.doubleValue(i) - y.doubleValue(i);
            sum += diff * diff;
        }
        return sum;
    }

    public static double squaredNorm(double[] array) {
        double sum = 0.0;
        for (double value : array) {
            sum += value * value;
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
    private static double inertiaFromLabels(NumberVector[] X, NumberVector[] centers, double[] labels) {
        double inertia = 0.0;
        for (int ee = 0; ee < X.length; ee++) {
            NumberVector x = X[ee];
            int label = (int) labels[ee];
            NumberVector center = centers[label];
            inertia += squaredNorm(x, center);
        }
        return inertia;
    }

    public static double vonMisesFisherLogPDF(NumberVector x, NumberVector mu, double kappa, int dimensionality) {
        double dotProduct = dotProduct(mu, x);
        double normalizationConstant = computeNormalizationConstant(kappa, dimensionality);
        double logPDF = Math.log(normalizationConstant) + kappa * dotProduct;
        return logPDF;
    }

    public static double computeNormalizationConstant(double kappa, int dimensionality) {
        double modifiedBessel = BesselJ.value(dimensionality / 2 - 1, kappa);
        return Math.pow(kappa, dimensionality / 2 - 1) / (Math.pow(2 * Math.PI, dimensionality / 2) * modifiedBessel);
    }




    /**
   * Performs the EM clustering algorithm on the given database.
   * <p>
   * Finally a hard clustering is provided where each clusters gets assigned the
   * points exhibiting the highest probability to belong to this cluster. But
   * still, the database objects hold associated the complete probability-vector
   * for all models.
   * 
   * @param relation Relation
   * @return Clustering result
   */
  public Clustering<M> run(Relation<V> relation) {
    //TODO
    return null;
  }

    @Override
    public TypeInformation[] getInputTypeRestriction() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getInputTypeRestriction'");
    }




    /**
   * Parameterization class.
   */
  public static class Par<V extends NumberVector, M extends MeanModel> implements Parameterizer {
    /**
     * Parameter to specify the number of clusters to find.
     */
    public static final OptionID K_ID = new OptionID("vmf.k", "The number of clusters to find.");

    /**
     * Parameter to specify the termination criterion 
     */
    public static final OptionID DELTA_ID = new OptionID("mf.delta", //
        "TODO");

    /**
     * Parameter to specify a minimum number of iterations.
     */
    public static final OptionID MINITER_ID = new OptionID("vmf.miniter", "Minimum number of iterations.");

    /**
     * Parameter to specify the maximum number of iterations.
     */
    public static final OptionID MAXITER_ID = new OptionID("vmf.maxiter", "Maximum number of iterations.");

    /**
     * Parameter to specify the saving of soft assignments
     */
    public static final OptionID SOFT_ID = new OptionID("vmf.soft", "Retain soft assignment of clusters.");

      /**
   * Parameter to specify the cluster center initialization.
   */
  static final OptionID INIT_ID = new OptionID("em.centers", "Method to choose the initial cluster centers.");

    /**
     * Number of clusters.
     */
    protected int k;

    /**
     * Stopping threshold
     */
    protected double delta;

    /**
     * Minimum number of iterations.
     */
    protected int miniter = 1;

    /**
     * Maximum number of iterations.
     */
    protected int maxiter = -1;

    /**
     * Retain soft assignments?
     */
    boolean soft = false;

      /**
   * Class to choose the initial means
   */
  protected KMeansInitialization initializer;

    @Override
    public void configure(Parameterization config) {
      new IntParameter(K_ID) //
          .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
          .grab(config, x -> k = x);
      new DoubleParameter(DELTA_ID, 1e-7)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE) //
          .grab(config, x -> delta = x);
      new IntParameter(MINITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> miniter = x);
      new IntParameter(MAXITER_ID)//
          .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_INT) //
          .setOptional(true) //
          .grab(config, x -> maxiter = x);
        new ObjectParameter<KMeansInitialization>(INIT_ID, KMeansInitialization.class, RandomlyChosen.class) //
          .grab(config, x -> initializer = x);
      new Flag(SOFT_ID) //
          .grab(config, x -> soft = x);
    }

    @Override
    public moVMF<V, M> make() {
      return new moVMF(k, miniter, maxiter, delta, soft, initializer);
    }
  }
}

