package com.example.refresh_selection;


import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.modelimport.keras.KerasLayer;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.jetbrains.annotations.NotNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Random;
import java.util.UUID;

import kotlin.jvm.internal.Intrinsics;


public class DeepFM implements FederatedModel{
    private ComputationGraph model;

    private static Logger log = LoggerFactory.getLogger(DeepFM.class);

    private String train_data_path = "/storage/self/primary/Download/data_balance/client1_train/";
    private String test_data_path = "/storage/self/primary/Download/data_balance/test/";


    private DataSetIterator AcitivityTrain;
    private DataSetIterator AcitivityTest;

//    public DeepFM(int N_SAMPLES_CLIENT_TRAINING, int N_SAMPLE_CLIENT_TEST) throws IOException {
//        AcitivityTrain = getDataSetIterator(train_data_path, N_SAMPLES_CLIENT_TRAINING);
////        AcitivityTest = getTestDataSetIterator(test_data_path, N_SAMPLE_CLIENT_TEST);
//    }

    class TensorsSum extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return  layerInput.sum("tensors_sum-" + UUID.randomUUID().toString(),false,1);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    class TensorsSquare extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return layerInput.mul("tensor_square-" + UUID.randomUUID().toString(),layerInput);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    class Lambda1 extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            return layerInput.mul("lambda1-" + UUID.randomUUID().toString(),0.5);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    class TensorMean extends SameDiffLambdaLayer {

        @Override
        public SDVariable defineLayer(SameDiff sameDiff, SDVariable layerInput) {
            if(this.layerName.equals("concat_embed_2d") || this.layerName.equals("cat_embed_2d_genure_mean"))
                return layerInput.mean("mean_pooling-" + UUID.randomUUID().toString(),true,1);
            else
                return layerInput.mean("mean_pooling-" + UUID.randomUUID().toString(),false,1);
        }

        @Override
        public InputType getOutputType(int layerIndex, InputType inputType) {
            return inputType;
        }
    }

    @Override
    public void buildModel(String modelzip_path) {

        KerasLayer.registerLambdaLayer("sum_of_tensors",  new TensorsSum());
        KerasLayer.registerLambdaLayer("square_of_tensors",  new TensorsSquare());
        KerasLayer.registerLambdaLayer("lambda_2",  new Lambda1());
        KerasLayer.registerLambdaLayer("cat_embed_2d_genure_mean", new TensorMean());
        KerasLayer.registerLambdaLayer("embed_1d_mean",  new TensorMean());

        //Load the model
        try {
            File modelzip = new File(modelzip_path+ "/MyMultiLayerNetwork.zip");
            System.out.print(modelzip);
            model = ModelSerializer.restoreComputationGraph(modelzip);
//            MultiLayerConfiguration neural_config2 = new NeuralNetConfiguration.Builder()
//                    .list()
//                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                            .nIn(10)
//                            .nOut(5)
//                            .activation(Activation.SOFTMAX)
//                            .build())
//                    .build();
//            MultiLayerNetwork model2 = new MultiLayerNetwork(neural_config2);
//            model2.init();
//
//            INDArray para1_W = model.getOutputLayer().getParam("W");
//            INDArray para1_b = model.getOutputLayer().getParam("b");
//
//            model2.getLayer(0).setParam("W", para1_W);
//            model2.getLayer(0).setParam("b", para1_b);
//
//            Layer[] layers = new Layer[model.getnLayers()];
//            for(int i = 0; i < model.getnLayers() - 1; i++) {
//                layers[i] = model.getLayer(i);
//            }
//            layers[layers.length-1] = model2.getLayer(0);
//            model.setLayers(layers);
//            model.init();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void train(int numEpochs) throws InterruptedException {

    }

    public String eval() {
        Evaluation model_eval = model.evaluate(AcitivityTest);
        return Double.toString(model_eval.accuracy()) + "," + Double.toString(model_eval.f1());
    }

    @Override
    public void saveModel(String modelName) {

    }

    @Override
    public void saveSerializeModel(String modelName) {

    }

    @Override
    public void uploadTo(String upload_path, String upload_url) throws IOException {

    }

    @Override
    public DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException {
        File train_data = new File(folderPath);
        FileSplit train = new FileSplit(train_data, NativeImageLoader.ALLOWED_FORMATS, new Random(123));
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(112, 112, 3, labelMaker);

        recordReader.initialize(train);
        recordReader.setListeners(new LogRecordListener());

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, 16, 1, 5);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        return dataIter;
    }
}
