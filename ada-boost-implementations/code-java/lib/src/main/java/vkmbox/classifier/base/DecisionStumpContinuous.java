package vkmbox.classifier.base;

import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 *  DecisionStump-Continuous classifier.
 *  Classifies: if value > threshold then sign else -1 * sign, 
 *  where sign = {-1,+1}
 */
public class DecisionStumpContinuous {
    
    private final int featureNumber;
    private final int sign;
    private final double threshold;
    
    public DecisionStumpContinuous(int featureNumber, int sign, double threshold) {
        this.featureNumber = featureNumber;
        this.sign = sign;
        this.threshold = threshold;
    }

    public int[] classify(INDArray dataX) {
        return getClassification(dataX, featureNumber, threshold, sign);
    }

    public static int[] getClassification(INDArray dataX, int featureNumber
            , double threshold, int sign) {
        double[] featureX = ArrayUtil.flattenDoubleArray(dataX.getColumn(featureNumber));
        int[] result = new int[featureX.length];
        for (int idx = 0; idx < featureX.length; idx++) {
            result[idx] = featureX[idx] > threshold ? sign: -1*sign;
        }
        return result; //[sign if value > threshold else -sign for value in X[:, feature_number]]
    }

    public static double getError(double[] featureX, int[] dataY, double[] weights
            , int sign, double threshold) {
        double error = 0.0;
        for (int idx = 0; idx < featureX.length; idx++) {
            int value = featureX[idx] > threshold ? sign: -1*sign;
            error += Math.abs((value-dataY[idx])/2)*weights[idx];
        }
        return error;
    }

    @Override
    public String toString() {
        return String.format("feature_number: %d, sign: %d, threshold: %f"
                , featureNumber, sign, threshold);
    }
}
