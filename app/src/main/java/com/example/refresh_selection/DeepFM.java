package com.example.refresh_selection;


import android.util.Log;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.CenterLossOutputLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.learning.config.NoOp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;
import java.util.UUID;

import kotlin.jvm.internal.Intrinsics;


public class DeepFM implements FederatedModel{
    private static final String TAG = "DeepFM";
    private static final int BATCH_SIZE = 2048;
    private static final int N_EPOCHS = 1;
    private static final int rngSeed = 42;

//    private static final int HEIGHT = 224;
//    private static final int WIDTH = 224;
//    private static final int OUTPUT_NUM = 5;

    private ComputationGraph model;

    private static Logger log = LoggerFactory.getLogger(DeepFM.class);

    private String train_data_path = "/storage/self/primary/Download/data_balance/client1_train/";
    private String test_data_path = "/storage/self/primary/Download/data_balance/test/";


    private DataSetIterator AcitivityTrain;
    private DataSetIterator AcitivityTest;

    public DeepFM(int N_SAMPLES_CLIENT_TRAINING, int N_SAMPLE_CLIENT_TEST) throws IOException {
        AcitivityTrain = getDataSetIterator(train_data_path, N_SAMPLES_CLIENT_TRAINING);
//        AcitivityTest = getTestDataSetIterator(test_data_path, N_SAMPLE_CLIENT_TEST);
    }


    @Override
    public void buildModel(String modelzip_path) {

        //Load the model
        try {
            File modelzip = new File(modelzip_path+ "/MyMultiLayerNetwork_beta6.zip");
            System.out.print(modelzip);
            model = ModelSerializer.restoreComputationGraph(modelzip);
            model.init();

//            Nd4j.getRandom().setSeed(12345);
//            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder().seed(12345)
//                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                    .updater(new Adam())
//                    .graphBuilder()
//                    .addInputs("input1")
//                    .addLayer("l1", new DenseLayer.Builder().nIn(4).nOut(5).activation(Activation.RELU).build(),
//                            "input1")
//                    .addLayer("lossLayer", new CenterLossOutputLayer.Builder()
//                            .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(5).nOut(numLabels)
//                            .lambda(lambda).activation(Activation.SOFTMAX).build(), "l1")
//                    .setOutputs("lossLayer").build();
//
//            ComputationGraph graph = new ComputationGraph(conf);
//            graph.init();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void train(int numEpochs) throws InterruptedException {
        Log.d(TAG, " start fit!");
        model.fit(AcitivityTrain, numEpochs);
    }

    public String eval() {
        Evaluation model_eval = model.evaluate(AcitivityTest);
        return Double.toString(model_eval.accuracy()) + "," + Double.toString(model_eval.f1());
    }

    @Override
    public void saveModel(String modelName) {
        try {
            File save_model = new File(modelName);
            model.save(save_model);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void saveSerializeModel(String modelName) {
        try {

            int layer_length = model.getNumLayers();
            JSONObject para_json = new JSONObject();
            for(int i = 0; i < layer_length; i++) {
                if(model.getLayer(i).getParam("W") != null) {
                    // 1. W param
                    JSONArray data_W = new JSONArray();
                    INDArray param_w = model.getLayer(i).getParam("W");
                    long[] param_shape_w = param_w.shape();

                    int total_size = 1;
                    for(int j = 0; j < param_shape_w.length; j++) {
                        total_size *= param_shape_w[j];
                    }
                    INDArray reshape_param = param_w.reshape(1, total_size);
                    for (int k = 0; k < reshape_param.getRow(0).length(); k++) {
                        data_W.put(reshape_param.getRow(0).getFloat(k));
                    }

                    // 2. b param
                    JSONArray data_b = new JSONArray();
                    INDArray param_b = model.getLayer(i).getParam("b");

                    for (int k = 0; k < param_b.columns(); k++) {
                        data_b.put(param_b.getRow(0).getFloat(k));
                    }

                    para_json.put(Integer.toString(i) + "_W", data_W);
                    para_json.put(Integer.toString(i) + "_b", data_b);
                }
            }

            FileWriter file = new FileWriter("/storage/self/primary/Download/save_weight/" + modelName);
            file.write(para_json.toString());
            file.flush();
            file.close();
        } catch (IOException | JSONException e) {
            e.printStackTrace();
        }
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
