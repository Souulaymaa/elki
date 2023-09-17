package elki.clustering.em;

import static elki.math.linearalgebra.VMath.*;

import elki.clustering.ClusteringAlgorithm;
import elki.clustering.em.models.EMClusterModel;
import elki.clustering.kmeans.initialization.KMeansInitialization;
import elki.clustering.kmeans.initialization.RandomlyChosen;
import elki.data.Cluster;
import elki.data.Clustering;
import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.model.EMModel;
import elki.data.model.MeanModel;
import elki.data.model.Model;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDataStore;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDUtil;
import elki.database.ids.ModifiableDBIDs;
import elki.database.relation.Relation;
import elki.distance.CosineDistance;
import elki.distance.NumberVectorDistance;
import elki.logging.Logging;
import elki.logging.statistics.DoubleStatistic;
import elki.result.Metadata;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.Flag;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import net.jafama.FastMath;
//import elki.clustering.em.models.EMClusterModelFactory;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class MoVMF<V extends NumberVector, M extends Model> implements ClusteringAlgorithm<Clustering<M>>{

    /**
   * Class to choose the initial means
   */
    protected KMeansInitialization initializer;

    protected NumberVectorDistance<? super V> distance = CosineDistance.STATIC;

    /**
     * The logger for this class.
     */
    private static final Logging LOG = Logging.getLogger(MoVMF.class);

    //protected EMClusterModelFactory<? super V, M> mfactory;

    //protected double prior = 0.;

    private int k;
    private int maxIterations;
    private int minIter;
    private double delta;
    //private int dimension;

    /**
     * Constructor
     * @param k number of components
     * @param maxIterations maximum number of iterations
     * @param minIter minimum number of Iterations
     * @param soft decides if it's a soft or hard clustering
     */

    public MoVMF(int k, int minIter, int maxIterations, double delta, boolean soft, KMeansInitialization initializer){
        super();
        this.k = k;
        this.maxIterations = maxIterations;
        this.minIter = minIter;
        this.initializer = initializer;
        this.delta = delta;
        if(maxIterations == 0){ this.maxIterations = Integer.MAX_VALUE;}
        //this.mfactory = mfactory;
    }

    /**
     * A method to train the model
     * @param relation the relation
     * @param tolerance the tolerance
     * @param maxIter maximum number of iterations
     * @return
     */

    //method too long, will split it for more efficiency

    public Clustering<MeanModel> train (Relation<V> relation, double tolerance, int maxIter) {

        //start by initialising the centers using a helping method
        double[][] centers = initializer.chooseInitialMeans(relation, k, distance);

        //initialise the probabilities alpha
        double[] sWeights;
        sWeights = new double[k];
        Arrays.fill(sWeights, 1.0 / k);

        // initialise kappas
        double[] kappas = new double[k];
        Arrays.fill(kappas, 1.0);

        //List<? extends EMClusterModel<? super V, M>> models = mfactory.buildInitialModels(relation, k);
        WritableDataStore<double[]> posterior = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_HOT | DataStoreFactory.HINT_SORTED, double[].class);
        System.out.print(posterior);
        double[] newCenters = new double[0];
        for (int iter = 0; iter < maxIter; iter++) {
            double[][] centersPrev = centers.clone(); //TODO vermutlich nicht möglich

            // Expectation step
            double loglikelihood = expectation(relation, centers, sWeights, kappas, posterior);
            DoubleStatistic likestat = new DoubleStatistic(this.getClass().getName() + ".loglikelihood");
            LOG.statistics(likestat.setDouble(loglikelihood));

            //expectation(relation, centers, sWeights, kappas, posterior);

            // Maximization step
            maximization(relation, posterior, centers, sWeights, kappas);

            // Check convergence
            double tolcheck = squaredNorm(centersPrev, centers);
            if (tolcheck <= tolerance) {
                System.out.printf("Converged at iteration %d: center shift %e within tolerance %e%n", iter, tolcheck, tolerance);
               //break;
            }
        }

        // Compute labels
        // fill result with clusters and models
        List<ModifiableDBIDs> hardClusters = new ArrayList<>(k);
        for(int i = 0; i < k; i++) {
            hardClusters.add(DBIDUtil.newArray());
        }


        // provide a hard clustering
        for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
            hardClusters.get(argmax(posterior.get(iditer))).add(iditer);

        }

        Clustering<MeanModel> result = new Clustering<>();
        Metadata.of(result).setLongName("EM Clustering");

        // provide models within the result
        for(int i = 0; i < k; i++) {
            result.addToplevelCluster(new Cluster<>(hardClusters.get(i), new MeanModel(centers[i])));
        }
        posterior.destroy();
        return result;

    }


    /**
     *
     * @param relation the relation
     * @param centers
     * @param weights
     * @param concentrations
     * @param posterior
     * @return
     */

    private double expectation(Relation<V> relation, double[][] centers, double[] weights, double[] concentrations, WritableDataStore<double[]> posterior){
        double emSum = 0.;
        for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
            V vec = relation.get(iditer);
            double[] probs = posterior.get(iditer);
            probs = probs != null ? probs : new double[k];
            for(int i = 0; i < k; i++) {
                double v = Math.log(weights[i]) + vonMisesFisherLogPDF(vec, centers[i], concentrations[i], vec.getDimensionality());
                probs[i] = v;
            }
            final double logP = EM.logSumExp(probs);
            for(int i = 0; i < k; i++) {
                probs[i] = FastMath.exp(probs[i] - logP);
            }
            posterior.put(iditer, probs);
            // if(loglikelihoods != null) {
            //     loglikelihoods.put(iditer, logP);
            // }
            emSum += logP;
        }
        return emSum / relation.size();
    }

    /**
     *
     * @param relation
     * @param posterior
     * @param centers
     * @param forceWeights
     * @param kappas
     * @return
     */
    private void maximization(Relation<V> relation, WritableDataStore<double[]> posterior, double centers[][], double [] forceWeights, double[] kappas){
        int d = centers[0].length;
        clear(centers);
        double[] tmpmean = new double[d];
        double[] wsum = new double[k];
        double[] newKappa = new double[k];
        double circularMeans = 0.0;

        for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
            double[] clusterProbabilities = posterior.get(iditer);
            V vec = relation.get(iditer);
            for(int i = 0; i < k; i++) {
                final double prob = clusterProbabilities[i];
                if(prob > 1e-10) { 
                    wsum[i] += prob;
                    final double f = prob / wsum[i]; // Do division only once
                    // Compute new means
                    for(int l = 0; l < d; l++) {
                        tmpmean[l] = centers[i][l] + (vec.doubleValue(i) - centers[i][l]) * f;
                    }

                    // Compute new Kappa
                    if (wsum[i] > 0) { // Ensure that there are data points assigned to the cluster

                        for (int l = 0; l < d; l++) {
                            circularMeans += Math.pow(tmpmean[l], 2);
                        }
                        // Compute the new kappas
                        double newKappas = circularMeans * (d / (1.0 - circularMeans));
                        newKappa[i] = newKappas;
                    }

                    System.arraycopy(tmpmean, 0, centers[i], 0, d);
                }
            }
        }
        for (int i = 0; i<k; i++){
            forceWeights[i] = wsum[i] / relation.size();
            kappas[i] = newKappa[i] /  wsum[i];
            for (int j = 0; j<d; j++){
                centers[i][j] = centers[i][j] / wsum[i];
            }
            
        }
        // ENDE
        
        for(int i = 0; i < k; i++) {
        // MLE
            final double weight = wsum[i] / relation.size();
            //models.get(i).finalizeEStep(weight, prior);
        }
        // old
       // for (int cc = 0; cc < clusters; cc++) {
        //    double weightsSum = 0.0;
        //  double[] weightedSum = new double[arr[0].getDimensionality()];
        //    double[] weightedDotSum = new double[arr[0].getDimensionality()];
        // double concentrationSum = 0.0;

        //    for(int ee = 0; ee <nExamples; ee++){
        //        double weight = posterior[ee][cc];
        //        weightsSum += weight;
        //      addVectorsInPlace(toNumberVector(weightedSum), arr[ee], weight);
        //      addVectorsInPlace(toNumberVector(weightedDotSum), multiplyVectors(arr[ee], arr[ee]), weight);
        //      concentrationSum += weight * dotProduct(arr[ee], arr[ee]);
        //    }
        //   weights[cc] = forceWeights != null ? forceWeights[cc] : weightsSum / nExamples;
        //  concentrations[cc] = concentrationSum / weightsSum;

//            if (weightsSum != 0.0) {
        //              multiplyVectorInPlace(weightedSum, 1.0 / weightsSum);
            }
    //   if (concentrationSum != 0.0) {
    //          concentrations[cc] /= (features * weightsSum);
    //      }
    //      System.arraycopy(weightedSum, 0, centers[cc], 0, features);
    //  }

    //  return new NumberVector[] {centers, toNumberVector(weights), toNumberVector(concentrations)};
    //}


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

    public static double dotProduct(double[] x, NumberVector y) {
        double result = 0.0;
        for (int i = 0; i < y.getDimensionality(); i++) {
            result += x[i] * y.doubleValue(i);
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
    private static double squaredNorm(double[][] x, double[][] y) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            sum += squaredNorm(x[i], y[i]);
        }
        return sum;
    }

    private static double squaredNorm(double[] x, double[] y) {
        double sum = 0.0;
        int dimensionality = x.length;
        for (int i = 0; i < dimensionality; i++) {
            double diff = x[i] - y[i];
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


    // private static double inertiaFromLabels(NumberVector[] X, NumberVector[] centers, double[] labels) {
    //     double inertia = 0.0;
    //     for (int ee = 0; ee < X.length; ee++) {
    //         NumberVector x = X[ee];
    //         int label = (int) labels[ee];
    //         NumberVector center = centers[label];
    //         inertia += squaredNorm(x, center);
    //     }
    //     return inertia;
    // }

    public static double vonMisesFisherLogPDF(NumberVector x, double[] mu, double kappa, int dimensionality) {
        double dotProduct = dotProduct(mu, x);
        double normalizationConstant = computeNormalizationConstant(kappa, dimensionality);
        double logPDF = Math.log(normalizationConstant) + kappa * dotProduct;
        return logPDF;
    }

    public static double computeNormalizationConstant(double kappa, int dimensionality) {
        double modifiedBessel = computeModifiedBessel(dimensionality / 2 - 1, kappa);
        return Math.pow(kappa, dimensionality / 2 - 1) / (Math.pow(2 * Math.PI, dimensionality / 2) * modifiedBessel);
    }

    public static double computeModifiedBessel(int order, double x)  {

        final double ACC = 40.0;
        final double BIGNO = 1.0e10;
        final double BIGNI = 1.0e-10;
        int j;
        double bi, bim, bip, tox, ans;

        if(order<0){return Double.NaN;}
        if(order == 0){return bessi0(x);}
        if(order == 1){return bessi1(x);}

        if (x == 0.0) {
            return 0.0;
        } else {
            tox = 2.0 / Math.abs(x);
            bip = ans = 0.0;
            bi = 1.0;
            for (j = 2 * (order + (int) Math.sqrt(ACC * order)); j > 0; j--) {
                bim = bip + j * tox * bi;
                bip = bi;
                bi = bim;
                if (Math.abs(bi) > BIGNO) {
                    ans *= BIGNI;
                    bi *= BIGNI;
                    bip *= BIGNI;
                }
                if (j == order) {
                    ans = bip;
                }
            }
            ans *= bessi0(x) / bi;
            return (x < 0.0 && order % 2 == 1) ? -ans : ans;
        }
    }

    public static double bessi0(double x) {
        double ax, ans;
        double y;

        ax = Math.abs(x);
        if (ax < 3.75) {
            y = x / 3.75;
            y = y * y;
            ans = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492
                    + y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))));
        } else {
            y = 3.75 / ax;
            ans = (Math.exp(ax) / Math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1
                    + y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2
                    + y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1
                    + y * 0.392377e-2))))))));
        }
        return ans;
    }

    public static double bessi1(double x) {
        double ax, ans;
        double y;

        ax = Math.abs(x);
        if (ax < 3.75) {
            y = x / 3.75;
            y = y * y;
            ans = ax * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (0.15084934
                    + y * (0.2658733e-1 + y * (0.301532e-2 + y * 0.32411e-3))))));
        } else {
            y = 3.75 / ax;
            ans = 0.2282967e-1 + y * (-0.2895312e-1 + y * (0.1787654e-1
                    - y * 0.420059e-2));
            ans = 0.39894228 + y * (-0.3988024e-1 + y * (-0.362018e-2
                    + y * (0.163801e-2 + y * (-0.1031555e-1 + y * ans))));
            ans *= (Math.exp(ax) / Math.sqrt(ax));
        }
        return x < 0.0 ? -ans : ans;
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
  public Clustering<MeanModel> run(Relation<V> relation) {
    if(relation.size() == 0) {
        throw new IllegalArgumentException("database empty: must contain elements");
    }
    return train(relation, delta, maxIterations);
  }

    @Override
    public TypeInformation[] getInputTypeRestriction() {
        return TypeUtil.array(distance.getInputTypeRestriction());
    }




    /**
   * Parameterization class.
   */
  public static class Par<V extends NumberVector, M extends Model> implements Parameterizer {
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

        /**
         * Cluster model factory.
         */
        //protected EMClusterModelFactory<V, M> mfactory;


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
    public MoVMF<V, M> make() {
      return new MoVMF<>(k, miniter, maxiter, delta, soft, initializer);
    }
  }
}

