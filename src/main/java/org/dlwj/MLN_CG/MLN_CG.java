package org.dlwj.MLN_CG;


import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 *  MultiLayerNetwork and Computation Graph
 *  MultiLayerNetwork - MultiLayerNetwork and Computation Graph
 *  Computation Graph - used for constructing networks with a more complex architecture than MLN
 **/
public class MLN_CG {

    public static void main( String[] args ) {

        Configurations configurations = new Configurations();

        //  Build MLN
        MultiLayerNetwork network = new MultiLayerNetwork( configurations.getMLNConfiguration() );

        //  Build CG
        ComputationGraph computationGraph = new ComputationGraph( configurations.getCGConfigurations() );

        //  Recurrent Network with skip connection
        ComputationGraph rnnNetwork = new ComputationGraph( configurations.recurrentNetworkConfiguration() );

    }

}
