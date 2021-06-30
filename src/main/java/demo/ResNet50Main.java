package demo;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.conf.graph.PreprocessorVertex;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.zoo.model.ResNet50;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

public class ResNet50Main {

	public static void main(String[] args) throws Exception {

		ComputationGraph initializedModel = (ComputationGraph) ResNet50.builder().build().initPretrained();

		INDArray input = new NativeImageLoader(224, 224).asMatrix("D:\\test.jpg");

		INDArray output = initializedModel.outputSingle(input);

		System.out.println(output);

		initializedModel = new TransferLearning.GraphBuilder(initializedModel).removeVertexAndConnections("fc1000")
				.addLayer("fc1000",
						new DenseLayer.Builder().activation(Activation.SOFTMAX).gainInit(0).dropOut(1).nIn(2048)
								.nOut(1000).weightInit(WeightInit.VAR_SCALING_UNIFORM_FAN_AVG).build(),
						"flatten_1")
				.setOutputs("flatten_1").build();

		output = initializedModel.outputSingle(input);

		System.out.println(output);

	}

}
