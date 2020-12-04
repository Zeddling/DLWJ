package org.dlwj.MLN_CG;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class Configurations {

    public MultiLayerConfiguration getMLNConfiguration() {
        //  Build MLN configuration
        return new NeuralNetConfiguration.Builder()
                .l2( 1e-4 )
                .seed( 123 )    //  For keeping the network outputs reproducible during runs by initializing weights and other network randomizations through a seed
                .updater( new Nesterovs( 0.1, 0.9 ))    //  algo for upfating parameters.
                .list()
                // .layer( 0, new DenseLayer.Builder().dropOut( 0.8 ).build() ) dropout layer
                //  .layer(0, new DenseLayer.Builder().biasInit(0).build())
                .layer( 0, new DenseLayer.Builder()
                        .nIn( 784 )
                        .nOut( 100 )
                        .weightInit( WeightInit.XAVIER )
                        .activation( Activation.RELU )
                        .build())
                .layer(1, new OutputLayer.Builder( LossFunctions.LossFunction.XENT )
                        .nIn( 100 )
                        .nOut( 10 )
                        .weightInit( WeightInit.XAVIER )
                        .activation( Activation.SIGMOID )
                        .build())
                .build();
    }

    public ComputationGraphConfiguration getCGConfigurations() {
        //  Build CG configuration
        return new NeuralNetConfiguration.Builder()
                .seed( 123 )
                .updater( new Nesterovs( 0.1, 0.9 ) )
                .graphBuilder()
                .addInputs( "input" )
                .addLayer( "L1", new DenseLayer.Builder()
                        .nIn( 3 )
                        .nOut( 4 )
                        .build(), "input" )
                .addLayer( "out1", new OutputLayer.Builder()
                        .lossFunction( LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD )
                        .nIn( 4 )
                        .nOut( 3 )
                        .build(), "L1" )
                .addLayer( "out2", new OutputLayer.Builder()
                        .lossFunction( LossFunctions.LossFunction.MSE )
                        .nIn( 4 )
                        .nOut( 2 )
                        .build(), "L1" )
                .setOutputs( "out1", "out2" )
                .build();
    }

    public ComputationGraphConfiguration recurrentNetworkConfiguration() {
        return new NeuralNetConfiguration.Builder()
                .graphBuilder()
                .addInputs( "inputs" )
                .addLayer( "L1", new LSTM.Builder()
                        .nIn( 5 )
                        .nOut( 5 )
                        .build(), "inputs" )
                .addLayer( "L2", new RnnOutputLayer.Builder()
                        .nIn( 5+5 )
                        .nOut( 5 )
                        .build(), "input", "L1" )
                .setOutputs( "L2" )
                .build();
    }

}
