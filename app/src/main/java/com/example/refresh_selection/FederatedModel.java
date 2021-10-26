package com.example.refresh_selection;

import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;

public interface FederatedModel {

    void buildModel(String modelsip_path);

    // void train(TrainerDataSource trainerDataSource);
    void train(int numEpochs) throws InterruptedException;

    String eval();

    void saveModel(String modelName);

    void saveSerializeModel(String modelName);

    void uploadTo(String upload_path, String upload_url) throws IOException;

    DataSetIterator getDataSetIterator(String folderPath, int nSamples) throws IOException;
}
