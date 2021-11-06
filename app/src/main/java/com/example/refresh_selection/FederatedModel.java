package com.example.refresh_selection;

import org.deeplearning4j.datasets.iterator.AsyncMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;

import java.io.IOException;

public interface FederatedModel {

    void buildModel(String modelsip_path);

    // void train(TrainerDataSource trainerDataSource);
    void train(int numEpochs) throws InterruptedException;

    String eval();

    void saveModel(String modelName);

    void saveSerializeModel(String modelName);

    void uploadTo(String upload_path, String upload_url) throws IOException;

    ListDataSetIterator getDataSetIterator(String filePath) throws IOException;
}
