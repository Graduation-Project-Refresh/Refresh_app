package com.example.webapi;

import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.SameDiffLambdaLayer;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.UUID;

public class TensorMean extends SameDiffLambdaLayer {

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