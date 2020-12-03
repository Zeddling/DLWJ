package org.dlwj.MNISTClassification;

import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;


public class MNISTApp {

    public static void main( String[] args ) throws IOException {

        int batchSize = 128;
        EmnistDataSetIterator.Set emnistSet = EmnistDataSetIterator.Set.BALANCED;
        EmnistDataSetIterator emnistTrain = new EmnistDataSetIterator( emnistSet, batchSize, true );
        EmnistDataSetIterator emnistTest = new EmnistDataSetIterator( emnistSet, batchSize, false );
        int outputNum = EmnistDataSetIterator.numLabels( emnistSet );
        int rngSeed = 123;
        int numRows = 28;
        int numColumns = 28;

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed( rngSeed )
                .updater( new Adam() )
                .l2( 1e-4 )
                .list()
                .layer( new DenseLayer.Builder()
                    .nIn( numRows * numColumns )
                    .nOut( 1000 )
                    .activation( Activation.RELU )
                    .weightInit( WeightInit.XAVIER )      //  Weight Initialization
                        .build())
                .layer( new OutputLayer.Builder( LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD )
                    .nIn( 1000 )
                    .nOut( outputNum )
                    .activation( Activation.SOFTMAX )
                    .weightInit( WeightInit.XAVIER )
                    .build())
                .build();

        //  Create Multilayer network
        MultiLayerNetwork network = new MultiLayerNetwork( configuration );
        network.init();

        //  Pass training listener that reports score every 10 iterations
        int eachIterations = 10;
        network.addListeners( new ScoreIterationListener( eachIterations ) );

        //  Fit dataset
        network.fit( emnistTrain, 2 );

        //  Evaluate basic performance
        Evaluation evaluation = network.evaluate( emnistTest );
        System.out.println( evaluation.stats() );
    }

}
