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
import elki.data.SparseNumberVector;
import elki.data.VectorUtil;
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
import elki.math.linearalgebra.VMath;
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

        WritableDataStore<double[]> posterior = DataStoreUtil.makeStorage(relation.getDBIDs(), DataStoreFactory.HINT_HOT | DataStoreFactory.HINT_SORTED, double[].class);
        System.out.print(posterior);
        double oldlikelihood = Double.NEGATIVE_INFINITY;
        for (int iter = 0; iter < maxIter; iter++) {
            // Expectation step
            double loglikelihood = expectation(relation, centers, sWeights, kappas, posterior);
            DoubleStatistic likestat = new DoubleStatistic(this.getClass().getName() + ".loglikelihood");
            LOG.statistics(likestat.setDouble(loglikelihood));

            // Maximization step
            maximization(relation, posterior, centers, sWeights, kappas);

            double likediff = loglikelihood - oldlikelihood;
            if ( iter > minIter &&  likediff <= tolerance) {
                System.out.printf("Converged at iteration %d: changedlikelihood %e within tolerance %e%n", iter, likediff, tolerance);
                break;
            }
            oldlikelihood = loglikelihood;
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
            double P = 0.;
            for(int i = 0; i < k; i++) {
                double v =  vonMisesFisherLogPDF(vec, centers[i], concentrations[i], vec.getDimensionality());
                probs[i] = v;
                P += v; //*  weights[i]
            }
            for(int i = 0; i < k; i++) {
                probs[i] = (probs[i]) / P; // TODO wsum?  * weights[i]
            }
            posterior.put(iditer, probs);
            emSum +=P;
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
        double[] wsum = new double[k];

        for(DBIDIter iditer = relation.iterDBIDs(); iditer.valid(); iditer.advance()) {
            double[] clusterProbabilities = posterior.get(iditer);
            V vec = relation.get(iditer);
            for(int i = 0; i < k; i++) {
                final double prob = clusterProbabilities[i];
                if(prob > 0) {
                    wsum[i] += prob;
                    // Compute new means
                    plusTimesIP(centers[i], vec, prob);
                }
            }
        }

        for (int i = 0; i<k; i++){
            if (wsum[i] > 0) { // Ensure that there are data points assigned to the cluster
                double circularMeans = 0.0;
                for (int l = 0; l < d; l++) {
                    circularMeans += Math.pow(centers[i][l], 2);
                }
                double r = Math.sqrt(circularMeans) / wsum[i];
                // Compute the new kappas
                double newKappa = ((r*d) - Math.pow(r, 3)) / (1.0 - Math.pow(r,2));
                kappas[i] = newKappa;
            }  
            VMath.normalizeEquals(centers[i]);
            forceWeights[i] = wsum[i] /= relation.size();
        }
    }

    private void plusTimesIP(double[] sum, NumberVector vec, double s){
        if(vec instanceof SparseNumberVector) {
            SparseNumberVector svec = (SparseNumberVector) vec;
            for(int j = svec.iter(); svec.iterValid(j); j = svec.iterAdvance(j)) {
                sum[svec.iterDim(j)] += svec.iterDoubleValue(j) * s;
            }
        }
        else {
            for(int j = 0; j < vec.getDimensionality(); j++) {
                sum[j] += vec.doubleValue(j) * s;
            }
        }
    }

    public static double vonMisesFisherLogPDF(NumberVector x, double[] mu, double kappa, int dimensionality) {
        double dotProduct = VectorUtil.dot(x, mu);
        double normalizationConstant = computeNormalizationConstant(kappa, dimensionality);
        double logPDF = normalizationConstant * Math.exp( kappa * dotProduct) ;
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

