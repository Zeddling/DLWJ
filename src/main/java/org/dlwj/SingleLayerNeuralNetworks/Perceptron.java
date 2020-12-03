package org.dlwj.SingleLayerNeuralNetworks;

import org.dlwj.SingleLayerNeuralNetworks.util.GaussianDistribution;

import java.util.Random;

import static org.dlwj.SingleLayerNeuralNetworks.util.ActivationFunction.step;

public class Perceptron {
    private final int train_N;    //  size of training data
    private final int test_N;     //  size of test data
    private final int nIn;        //  dimensions of input data
    private double[] w;           //  weight vector

    private final int epochs;                             // maximum training epochs
    private final double learningRate;

    /*
     *  Create training data and test data for demo.
     *
     * Let training data set for each class follow Normal (Gaussian) distribution here:
     *      class 1 : x1 ~ N( -2.0, 1.0 ), y1 ~ N( +2.0, 1.0 )
     *      class 2 : x2 ~ N( +2.0, 1.0 ), y2 ~ N( -2.0, 1.0 )
     */
    private static final Random rng = new Random(1234);
    private final GaussianDistribution dist1 = new GaussianDistribution(-2.0, 1.0, rng );
    private final GaussianDistribution dist2 = new GaussianDistribution(2.0, 1.0, rng );

    public Perceptron ( int train_N, int test_N, int nIn, int epochs, int learningRate ) {
        this.train_N = train_N;
        this.test_N = test_N;
        this.nIn = nIn;
        this.epochs = epochs;
        this.learningRate = learningRate;
        w = new double[nIn];
    }

    public double[][] getTrainData() {
        double[][] X_train =  new double[train_N][nIn];      //  Input data for training
        {
            int i = 0;
            while ( i < ( train_N / 2 - 1 ) ) {        //  dataset in class 1
                X_train[i][0] = dist1.random();
                X_train[i][1] = dist2.random();
                i++;
            }
        }
        int i = train_N / 2;
        do {      //  dataset in class 2
            X_train[i][0] = dist2.random();
            X_train[i][1] = dist1.random();
            i++;
        } while ( i < train_N );
        return X_train;
    }

    public int[] setTrainDataLabel() {
        int[] trainLabels = new int[train_N];                    //  Label for training
        {
            int i = 0;
            while ( i < ( train_N / 2 - 1 ) ) {
                trainLabels[i] = 1;
                i++;
            }
        }
        int i = train_N / 2;
        while (i < train_N) {
            trainLabels[i] = -1;
            i++;
        }

        return trainLabels;
    }

    public double[][] getTestData() {
        double[][] X_test = new double[test_N][nIn];         //  Input data for testing
        {
            int i = 0;
            while (i < ( test_N / 2 - 1 )) {        //  dataset in class 1
                X_test[i][0] = dist1.random();
                X_test[i][1] = dist2.random();
                i++;
            }
        }
        int i = test_N / 2;
        while ( i < test_N ) {        //  dataset in class 1
            X_test[i][0] = dist2.random();
            X_test[i][1] = dist1.random();
            i++;
        }
        return X_test;
    }

    public int[] setTestDataLabel() {
        int[] testLabels = new int[test_N];                    //  Label for test
        {
            int i = 0;
            while (i < ( test_N / 2 - 1 ) ) {
                testLabels[i] = 1;
                i++;
            }
        }
        int i = test_N / 2;
        while ( i < test_N ) {
            testLabels[i] = -1;
            i++;
        }
        return testLabels;
    }

    public int train( double[] X, int t, double learningRate ) {
        int classified = 0;
        double c = 0;

        //  check if the data is classified correctly
        for ( int i = 0; i < nIn; i++ ) {
            c += w[i] * X[i] * t;
        }

        //  gradient descent if data is wrongly classified
        if (c < 0) {
            classified = 1;
        }   else {
            for ( int i = 0; i < nIn; i++ ) {
                w[i] += learningRate * X[i] * t;
            }
        }
        return classified;
    }

    public int predict ( double[] x ) {
        double preActivation = 0;

        for ( int i = 0; i < nIn; i++ ) {
            preActivation += w[i] * x[i];
        }
        return step(preActivation);
    }

    public double[] evaluate(int[] predictedLabels, int[] testLabels) {
        int[][] confusion_matrix = new int[2][2];
        double accuracy = 0;
        double precision = 0;
        double recall = 0;

        for ( int i = 0; i < test_N; i++ ) {
            if ( predictedLabels[i] > 0 ) {
                if ( testLabels[i] > 0 ) {
                    accuracy += 1;
                    precision += 1;
                    recall += 1;
                    confusion_matrix[0][0] += 1;
                } else {
                    confusion_matrix[1][0] += 1;
                }
            } else {
                if ( testLabels[i] > 0 ) {
                    confusion_matrix[0][1] += 1;
                } else {
                    accuracy += 1;
                    confusion_matrix[1][1] += 1;
                }
            }
        }
        accuracy /= test_N;
        precision /= confusion_matrix[0][0] + confusion_matrix[1][0];
        recall /= confusion_matrix[0][0] + confusion_matrix[0][1];
        return new double[]{accuracy, precision, recall};
    }





}
