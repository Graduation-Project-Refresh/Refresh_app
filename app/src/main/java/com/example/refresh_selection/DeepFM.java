package com.example.refresh_selection;


import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import org.datavec.api.split.FileSplit;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.impl.InputVertex;
import org.deeplearning4j.util.ModelSerializer;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Hashtable;

import java.util.List;
//import org.datavec.spark.functions.LineRecordReaderFunction;


public class DeepFM implements FederatedModel {
    private static final String TAG = "DeepFM";
    private static final int BATCH_SIZE = 50;
    private static final int labelIndex = 8;
    private static final int N_EPOCHS = 1;
    private static final int rngSeed = 42;

//    private static final int HEIGHT = 224;
//    private static final int WIDTH = 224;
//    private static final int OUTPUT_NUM = 5;

    private ComputationGraph model;

    private static Logger log = LoggerFactory.getLogger(DeepFM.class);

    private String train_data_path = "/storage/self/primary/Download/data_balance/client1_train/new_input.csv";
    private String test_data_path = "/storage/self/primary/Download/data_balance/test/";

    private String mlsfc_vocab_path = "mlsfc_vocab.txt";
    private String mcate_nm_path = "mcate_nm_vocab.txt";
    private String day_path = "day_vocab.txt";
    private String sex_path = "Sex_vocab.txt";

    private ListDataSetIterator AcitivityTrain;
    private DataSetIterator AcitivityTest;
    private Context context;

    public DeepFM(Context context) throws IOException {
        this.context = context;
        AcitivityTrain = getDataSetIterator(train_data_path);
//        AcitivityTest = getTestDataSetIterator(test_data_path, N_SAMPLE_CLIENT_TEST);
    }

    public Hashtable get_vocab(String file_name) {
        Hashtable<String, Integer> vocab = null;
        try {
            vocab = new Hashtable<String, Integer>();

            AssetManager as = context.getResources().getAssets();
            InputStream is = as.open(file_name);
            BufferedReader reader = new BufferedReader(new InputStreamReader(is));

            String str;
            while (true) {
                if (!((str = reader.readLine()) != null))
                    break;
                else {
                    String[] key_val = str.split("\t");
                    vocab.put(key_val[0], Integer.parseInt(key_val[1]));
                }
            }
        } catch(IOException e){
            e.printStackTrace();
        }
        return vocab;
    }


    @Override
    public void buildModel(String modelzip_path) {

        //Load the model
        try {
            File modelzip = new File(modelzip_path + "/MyMultiLayerNetwork_beta6.zip");
            System.out.print(modelzip);
            model = ModelSerializer.restoreComputationGraph(modelzip);
            model.init();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void train(int numEpochs) throws InterruptedException {
        Log.d(TAG, " start fit!");
        model.fit(AcitivityTrain);
    }

    public String eval() {
        Evaluation model_eval = model.evaluate(AcitivityTrain);
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
            for (int i = 0; i < layer_length; i++) {
                if (model.getLayer(i).getParam("W") != null) {
                    // 1. W param
                    JSONArray data_W = new JSONArray();
                    INDArray param_w = model.getLayer(i).getParam("W");
                    long[] param_shape_w = param_w.shape();

                    int total_size = 1;
                    for (int j = 0; j < param_shape_w.length; j++) {
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
    public ListDataSetIterator getDataSetIterator(String filePath) throws IOException {

        // txt 파일 읽어서 리스트나 dictionary로 저자해놔야함
        Hashtable mlsfc_vocab_table = get_vocab(mlsfc_vocab_path);
        Hashtable mcate_nm_table = get_vocab(mcate_nm_path);
        Hashtable day_table = get_vocab(day_path);
        Hashtable sex_table = get_vocab(sex_path);


        //Load the training data:
        RecordReader rr = new CSVRecordReader(1);
        AsyncMultiDataSetIterator dataIter = null;
        try {
            rr.initialize(new FileSplit(new File(filePath)));
            ArrayList originalData = new ArrayList();
            while (rr.hasNext()) {
                List line = rr.next();
                for (int i = 0; i < line.size(); i++) {
                    // encode
                    if(i == 3 || i == 4 || i ==5 || i == 8){
                        line.set(i, Integer.parseInt(line.get(i).toString()));
                    }else if( i == 0 ){
                        line.set(i, mlsfc_vocab_table.get(line.get(i).toString()));
                    }else if( i == 1 ){
                        line.set(i, mcate_nm_table.get(line.get(i).toString()));
                    }else if( i == 2){
                        line.set(i, sex_table.get(line.get(i).toString()));
                    }else if( i == 6 ){
                        line.set(i, day_table.get(line.get(i).toString()));
                    }else if( i == 7){
                        line.set(i, mcate_nm_table.get(line.get(i).toString()));
                    }
                }
                originalData.add(line);
            }

            int col_size = originalData.size();
            int row_size = ((ArrayList)(originalData.get(0))).size();

            float[][] mlsfc = new float[col_size][1];
            float[][] mcate_nm = new float[col_size][1];
            float[][] sex = new float[col_size][1];
            float[][] age = new float[col_size][1];
            float[][] month = new float[col_size][1];
            float[][] time = new float[col_size][1];
            float[][] day = new float[col_size][1];
            float[][] fav_plc = new float[col_size][1];
            float[][] clk = new float[col_size][1];

            for(int i = 0; i < col_size; i++){
                for(int j =0; j<row_size; j++){
                    switch (i){
                        case 0:
                            mlsfc[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 1:
                            mcate_nm[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 2:
                            sex[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 3:
                            age[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 4:
                            month[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 5:
                            time[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 6:
                            day[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 7:
                            fav_plc[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                        case 8:
                            clk[i][0] = Float.parseFloat(((ArrayList)originalData.get(i)).get(j).toString());
                            break;
                    }
                }
            }
            // 모델 빌드 (임시)
            File modelzip = new File("/storage/self/primary/Download/save_model/MyMultiLayerNetwork_beta6.zip");
            System.out.print(modelzip);
            model = ModelSerializer.restoreComputationGraph(modelzip);
//            model.init();

            INDArray mlsfc_ind = Nd4j.create(mlsfc);
            INDArray mcate_nm_ind = Nd4j.create(mcate_nm);
            INDArray sex_ind = Nd4j.create(sex);
            INDArray age_ind = Nd4j.create(age);
            INDArray month_ind = Nd4j.create(month);
            INDArray time_ind = Nd4j.create(time);
            INDArray day_ind = Nd4j.create(day);
            INDArray fav_plc_ind = Nd4j.create(fav_plc);
            INDArray clk_ind = Nd4j.create(clk);

            System.out.print(model.summary());
//            model.feedForward(new INDArray[]{age_ind, mlsfc_ind, mcate_nm_ind, sex_ind, month_ind, time_ind, day_ind, fav_plc_ind}, true);

//            model.fit(new INDArray[]{age_ind, mlsfc_ind, mcate_nm_ind, sex_ind, month_ind, time_ind, day_ind, fav_plc_ind}, new INDArray[]{clk_ind});
            MultiDataSet ds = new MultiDataSet(new INDArray[]{age_ind, mlsfc_ind, mcate_nm_ind, sex_ind, month_ind, time_ind, day_ind, fav_plc_ind}, new INDArray[]{clk_ind});
            model.setInputs(age_ind, mlsfc_ind, mcate_nm_ind, sex_ind, month_ind, time_ind, day_ind, fav_plc_ind);
            model.setLabels(clk_ind);
            model.fit(); // 모델 train

//
//            RecordReader recordReader = new CSVRecordReader(1);
//            recordReader.initialize(new FileSplit(new File("/storage/self/primary/Download/data_balance/client1_train/new_input_encoded.csv")));
//
//            DataSet multiDataSet = new DataSet(input, label);
//
////            multiDataSet.setFeatures(new INDArray[]{input.getColumn(0, true),input.getColumn(1, true),input.getColumn(2, true),input.getColumn(3, true),input.getColumn(4, true),input.getColumn(5, true),input.getColumn(6, true),input.getColumn(7, true)});
////            multiDataSet.setLabels(new INDArray[]{label});

////            dataIter = new ListDataSetIterator(multiDataSets, 5);
////            dataIter = new AsyncMultiDataSetIterator(iter,5);
////            model.fit(multiDataSet);
//
////            INDArray input = Nd4j.create(1,2);
////            INDArray output = Nd4j.create(1,1);
////            model.fit(new INDArray[]{input,input,input,input,input,input,input,input}, new INDArray[]{output});
//
////            INDArray input_ndarray = Nd4j.create(input_list);
////            INDArray input_ndarray2 = Nd4j.create(input_list[0]);
////            INDArray output_ndarray = Nd4j.create(output_list);
////            INDArray output_ndarray2 = Nd4j.create(output_list[0]);
////            INDArray[] input = {input_ndarray.getColumn(0), input_ndarray.getColumn(0), input_ndarray.getColumn(0),input_ndarray.getColumn(0),input_ndarray.getColumn(0),input_ndarray.getColumn(0),input_ndarray.getColumn(0),input_ndarray.getColumn(0)};
////            INDArray[] output = {output_ndarray.getColumn(0)};
////            DataSet dataSet = new DataSet(input_ndarray, output_ndarray);
////            MultiDataSet multiDataset = new MultiDataSet();
////            multiDataset.setFeatures(input);
////            multiDataset.setLabels(output);
////
////            dataIter = new ListDataSetIterator(listDs, 5);
////            DataSet t = dataIter.next();
////            System.out.print("Dataset : "+ t);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return null;
    }

    public void data_encode() {

//        LabelEncoder le = new LabelEncoder("module", "name");


    }
}
