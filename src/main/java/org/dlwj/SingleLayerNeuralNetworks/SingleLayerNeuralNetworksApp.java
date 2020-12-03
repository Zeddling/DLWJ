package org.dlwj.SingleLayerNeuralNetworks;

import org.dlwj.SingleLayerNeuralNetworks.Perceptron;

/**
 * Hello world!
 *
 */
public class SingleLayerNeuralNetworksApp {

    public static void main( String[] args ) {
        Perceptron perceptron = new Perceptron(1000, 200, 2, 2000, 1);
        int classified = 0;
        int epoch = 0;

        double[][] trainDataset = perceptron.getTrainData();
        int[] trainDataLabel = perceptron.setTrainDataLabel();

        double[][] testDataset = perceptron.getTestData();
        int[] testDataLabel = perceptron.setTestDataLabel();
        int[] predictedLabels = new int[200];

        //  Train model
        do {
            for ( int i = 0; i < 1000; i++ ) classified = perceptron.train(trainDataset[i], trainDataLabel[i], 1);
            epoch++;
        } while ( epoch < 2000 || classified == 1000 );

        //  Test model
        for ( int i = 0; i < 200; i++ ) {
            predictedLabels[i] = perceptron.predict( testDataset[i] );
        }

        //  Model evaluation
        double[] evaluation = perceptron.evaluate(predictedLabels, testDataLabel);
        System.out.println("Accuracy: " + evaluation[0]);
        System.out.println("Precision: " + evaluation[1]);
        System.out.println("Recall: " + evaluation[2]);

    }
}






