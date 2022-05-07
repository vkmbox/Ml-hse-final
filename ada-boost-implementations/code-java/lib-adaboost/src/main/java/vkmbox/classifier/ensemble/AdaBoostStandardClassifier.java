package vkmbox.classifier.ensemble;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import vkmbox.classifier.base.DecisionStumpContinuous;

import java.util.Map;
import java.util.List;
import java.time.Clock;
import java.time.ZoneId;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.ArrayList;

public class AdaBoostStandardClassifier {
    
    private static final int[] SIGNS = {-1, 1};
    private static final Clock CLOCK = Clock.system(ZoneId.systemDefault());
    
    private final int estimators;
    private final double tolerance;
    private final List<Double> ensembleAlphas = new ArrayList<>();
    private final List<DecisionStumpContinuous> ensembleClassifiers = new ArrayList<>();
    
    public AdaBoostStandardClassifier(int estimators, double tolerance) {
        this.estimators = estimators;
        this.tolerance = tolerance;
    }

    public AdaBoostStandardClassifier(int estimators) {
        this(estimators, 1e-10);
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY, double[] sampleWeights, boolean trace) {
        ensembleAlphas.clear();
        ensembleClassifiers.clear();
        long timeStart = CLOCK.millis();
        int samplesCount = dataX.rows();
        double[] weights = sampleWeights;
        if (Objects.isNull(sampleWeights)) {
            weights = new double[samplesCount];
            Arrays.fill(weights, 1.0/samplesCount);
        }
        Map<String, List> history = trace? new HashMap<>(): null;

        for (int dummy = 0; dummy < estimators; dummy++) { // in range(self.n_estimators)
            //INDArray weightsMulti = Nd4j.tile(weights, featuresCount); //(1, features_count)); //dd_t
            var optimum = getDecisionStump(dataX, dataY, weights);
            if (optimum.minimalError >= 0.5) {
                return FitResult.of("error_level_exceeded", history);
            }
            double alphaT = 0.5 * Math.log((1 - optimum.minimalError)/(optimum.minimalError+tolerance));
            ensembleAlphas.add(alphaT);
            ensembleClassifiers.add(optimum.decisionStump);
            log(trace, history, optimum.minimalError, weights, timeStart);
            if (optimum.minimalError == 0) {
                return FitResult.of("error_free_classifier_found", history);
            }
            int[] forecast = optimum.decisionStump.classify(dataX);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] *= Math.exp(-alphaT * dataY.getInt(sampleNumber) * forecast[sampleNumber]);
            }
            double weightSum = sum(weights);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                weights[sampleNumber] = weights[sampleNumber]/(weightSum+tolerance);
            }
        }
        return FitResult.of("iterations_exceeded", history);
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY) {
        return fitIND(dataX, dataY, null, false);
    }
    
    public FitResult fitIND(INDArray dataX, INDArray dataY, boolean trace) {
        return fitIND(dataX, dataY, null, trace);
    }
    
    public INDArray getEsembleResult(INDArray dataX) {
        int samplesCount = dataX.rows();
        //org.nd4j.linalg.factory.DataType aa;
        INDArray buffer = Nd4j.zeros(new int[]{samplesCount}); //, DataType.INT32);
        for (int index = 0; index < ensembleAlphas.size(); index++) {
            buffer.addi(ensembleClassifiers.get(index)
                .classifyIND(dataX, DataType.DOUBLE).mul(ensembleAlphas.get(index)));
        }
        return buffer;
    }

    public INDArray predictIND(INDArray dataX) {
        return Transforms.sign(getEsembleResult(dataX));
    }
    
    public int[] predict(INDArray dataX) {
        return predictIND(dataX).toIntVector();
    }
    
    public double getMarginL1(INDArray dataX) {
        INDArray buffer = getEsembleResult(dataX);
        double alphaModulo = ensembleAlphas.stream().mapToDouble(Double::doubleValue).sum() + tolerance;
        return buffer.aminNumber().doubleValue()/alphaModulo;
    }
    
    private StampResult getDecisionStump(INDArray dataX, INDArray dataY, double[] weights) {
        int samplesCount = dataX.rows(), featuresCount = dataX.columns();
        double minimalError = Double.MAX_VALUE, minimalThreshold = Double.MAX_VALUE;
        int minimalFeature = 0, minimalSign = 0;

        for (int featureNumber = 0; featureNumber < featuresCount; featureNumber++) {
            INDArray feature = dataX.getColumn(featureNumber);
            for (int sampleNumber = 0; sampleNumber < samplesCount; sampleNumber++) {
                double threshold = feature.getDouble(sampleNumber);
                for (int sign : SIGNS) {
                    double currentError = DecisionStumpContinuous.getError
                        (dataX, dataY, featureNumber, weights, sign, threshold);
                    if (minimalError > currentError) {
                        minimalError = currentError;
                        minimalFeature = featureNumber;
                        minimalSign = sign; 
                        minimalThreshold = threshold;
                    }
                }
            }
        }
        return StampResult.of(minimalError
                , new DecisionStumpContinuous(minimalFeature, minimalSign, minimalThreshold));
    }

    private void log(boolean trace, Map<String, List> history
            , double minimalError, double[] weights, long timeStart) {
        if (trace) {
            history.computeIfAbsent("time", arg -> new ArrayList<Long>()).add(CLOCK.millis() - timeStart);
            history.computeIfAbsent("error", arg -> new ArrayList<Double>()).add(minimalError);
            history.computeIfAbsent("d_t", arg -> new ArrayList<double[]>()).add(weights);
        }
    }
    
    public static double sum(double...values) {
        double result = 0;
        for (double value:values)
            result += value;
        return result;
    }

    /*def print_ensemble(self):
        value = ""
        for elm in self.ensemble:
            value += "alpha={}, classifier=<{}>;".format(elm[0], elm[1].str())
        return value
    */
    
    /*@Builder
    public static class EnsembleItem {
        private final double alpha;
        private final DecisionStumpContinuous classifier;
    }*/
    
    @Getter
    @RequiredArgsConstructor
    public static class StampResult {
        private final Double minimalError;
        private final DecisionStumpContinuous decisionStump;
        
        public static StampResult of(Double minimalError, DecisionStumpContinuous decisionStump) {
            return new StampResult(minimalError, decisionStump);
        }
    }
    
    @Getter
    @RequiredArgsConstructor
    public static class FitResult {
        private final String result;
        private final Map<String, List> history;
        
        public static FitResult of(String result, Map<String, List> history) {
            return new FitResult(result, history);
        }
    }

}
