package io.github.jhipster.sample.prediction;

import ai.djl.MalformedModelException;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.output.DetectedObjects;
import ai.djl.mxnet.zoo.MxModelZoo;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;


public class Inference {
    private static final Logger logger = LoggerFactory.getLogger(Inference.class);

    //values must match settings used during training
    //the number of classification labels: boots, sandals, shoes, slippers
    private static final int NUM_OF_OUTPUT = 4;

    //the height and width for pre-processing of the image
    private static final int NEW_HEIGHT = 100;
    private static final int NEW_WIDTH = 100;


    public DetectedObjects predictor(ByteArrayInputStream image) throws MalformedModelException, ModelNotFoundException, IOException, TranslateException {
        Image input = ImageFactory.getInstance().fromInputStream(image);
        BufferedImage img = (BufferedImage) input;

        // var imageFile = Paths.get("src/main/resources/new-york.jpg");
        //var img = BufferedImageUtils.fromFile(imageFile);

        ZooModel<Image, DetectedObjects> model =
            MxModelZoo.SSD.loadModel(new ProgressBar());

        var predictor = model.newPredictor().predict(input);
        ImageIO.write(img, "png", new File("new-york.png"));
        model.close();
        return predictor;
    }
}
