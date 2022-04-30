package vkmbox.classifier.ensemble;

import lombok.Builder;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.time.Clock;
import java.time.ZoneId;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import org.nd4j.common.primitives.Pair;

public class AdaBoostStandardClassifier {
    
    private static final int[] SIGNS = {-1, 1};
    private static final Clock CLOCK = Clock.system(ZoneId.systemDefault());
    
    private final int estimators;
    private final double tolerance;
    private final List<EnsembleItem> ensemble = new ArrayList<>();
    
    public AdaBoostStandardClassifier(int estimators, double tolerance) {
        this.estimators = estimators;
        this.tolerance = tolerance;
    }

    public AdaBoostStandardClassifier(int estimators) {
        this(estimators, 1e-10);
    }
    
    public FitResult fit(INDArray dataX, INDArray dataY, boolean trace, double[] sampleWeights) {
        ensemble.clear();
        long timeStart = CLOCK.millis();
        int sampleSize = dataX.rows(), featuresCount = dataX.columns();
        double[] dt = sampleWeights;
        if (Objects.isNull(sampleWeights)) {
            dt = new double[sampleSize];
            Arrays.fill(dt, 1/sampleSize);
        }
        INDArray weights = Nd4j.create(dt, new int[]{dt.length, 1});
        //yy = np.tile(np.array([y]).transpose(), (1, features_count))
        INDArray datayMulti = Nd4j.tile(dataY, featuresCount); //dataY
        Map<String, List> history = trace? new HashMap<>(): null;

        for (int dummy = 0; dummy < estimators; dummy++) { // in range(self.n_estimators)
            INDArray weightsMulti = Nd4j.tile(weights, featuresCount); //(1, features_count)); //dd_t
            Pair<Double, DecisionStumpContinuous> optimum = getDecisionStump(dataX, datayMulti, weightsMulti);
        }
        /*    minimal_error, decision_stump = self.get_decision_stump(X, yy, dd_t)
            if minimal_error >= 0.5:
                return 'error_level_exceeded', history
            alpha_t = 0.5 * np.log((1-minimal_error)/(minimal_error+self.tolerance))
            self.ensemble.append((alpha_t, decision_stump))
            self.log(trace, history, minimal_error, d_t, time_start)
            if minimal_error == 0:
                return 'error_free_classifier_found', history
            d_t = np.multiply(d_t, np.exp(-alpha_t * np.multiply(y, decision_stump.classify(X))))
            d_t = d_t/(np.sum(d_t)+self.tolerance)*/
            
        return FitResult.builder().result("iterations_exceeded").history(history).build();
    }
    
    public FitResult fit(INDArray dataX, INDArray dataY, boolean trace) {
        return fit(dataX, dataY, trace, null);
    }
    
    /*public int[] predict(self, X) {
        sample_size = X.shape[0]
        buffer = np.zeros(sample_size)
        for elm in self.ensemble:
            buffer += elm[0]*np.array(elm[1].classify(X))
            
        return np.sign(buffer)
    }*/
    
    private Pair<Double, DecisionStumpContinuous> getDecisionStump
            (INDArray dataX, INDArray dataY, INDArray weightsMulti) {
        int sampleSize = dataX.rows(), featuresCount = dataX.columns();
//        measurements, features_count = X.shape[0], X.shape[1]
        double minimalError = Double.MAX_VALUE; //, minimal_feature, minimal_sign, minimal_threshold \
            //= None, None, None, None

        ind = np.argsort(X, axis=0)
        xx = np.take_along_axis(X, ind, axis=0)
        yy = np.take_along_axis(yym, ind, axis=0)
        dd_t = np.take_along_axis(dd_tm, ind, axis=0)

        for feature_number in range(features_count):
            for threshold in range(measurements):
                error_plus, error_minus = 0.0, 0.0
                for pos in range(measurements):
                    if pos <= threshold:
                        error_plus += dd_t[pos, feature_number] if yy[pos, feature_number] == 1 else 0
                        error_minus += dd_t[pos, feature_number] if yy[pos, feature_number] == -1 else 0
                    else:
                        error_plus += dd_t[pos, feature_number] if yy[pos, feature_number] == -1 else 0
                        error_minus += dd_t[pos, feature_number] if yy[pos, feature_number] == 1 else 0

                if error_plus <= error_minus and (minimal_error is None or error_plus < minimal_error):
                    minimal_error, minimal_feature, minimal_sign, minimal_threshold \
                        = error_plus, feature_number, +1, xx[threshold, feature_number]
                if error_minus < error_plus and (minimal_error is None or error_minus < minimal_error):
                    minimal_error, minimal_feature, minimal_sign, minimal_threshold \
                        = error_minus, feature_number, -1, xx[threshold, feature_number]

        return minimal_error, DecisionStumpContinuous(minimal_feature, minimal_sign, minimal_threshold)
    }

    private void log(boolean trace, Map<String, List> history
            , double error, double[] weights, long timeStart) {
        if (trace) {
            history['time'].append((datetime.now() - time_start).total_seconds())
            history['error'].append(minimal_error)
            history['d_t'].append(d_t)
        }
    }

    def print_ensemble(self):
        value = ""
        for elm in self.ensemble:
            value += "alpha={}, classifier=<{}>;".format(elm[0], elm[1].str())
        return value
    */
    
    @Builder
    public static class EnsembleItem {
        private final double alpha;
        private final DecisionStumpContinuous classifier;
    }
    
    @Builder
    public static class FitResult {
        private final String result;
        private final Map<String, List> history;
    }

}
