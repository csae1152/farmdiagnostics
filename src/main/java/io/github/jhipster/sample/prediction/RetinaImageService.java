package io.github.jhipster.sample.prediction;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.BaseImageRecordReader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultipleEpochsIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

@Service
public class RetinaImageService {
    private static final Logger log = LoggerFactory.getLogger(RetinaImageService.class);
    protected static int height=32;
    protected static int width=32;

    protected static int channels = 3;
    protected static int batchSize=150;// tested 50, 100, 200
    protected static long seed = 123;
    protected static Random rng = new Random(seed);
    protected static int iterations = 1;
    protected static int nEpochs = 150; // tested 50, 100, 200
    protected static double splitTrainTest = 0.8;
    protected static boolean save = true;
    private int numLabels;

    MultiLayerNetwork model;

     public void init(MultipartFile file) throws IOException {
         /**
          * Setting up data
          */
         ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
         File mainPath = new File("/Users/helmu/OneDrive/Documents/java/farmdiagnostics/dataset");
         FileSplit fileSplit = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
         int numExamples = Math.toIntExact(fileSplit.length());
         numLabels = fileSplit.getRootDir().listFiles(File::isDirectory).length; //This only works if your root is clean: only label subdirs.
         BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

         /**
          * Split data: 80% training and 20% testing
          */
         InputSplit[] inputSplit = fileSplit.sample(pathFilter, splitTrainTest, 1 - splitTrainTest);
         InputSplit trainData = inputSplit[0];
         InputSplit testData = inputSplit[1];

         /**
          *  Create extra synthetic training data by flipping, rotating
          #  images on our data set.
          */
         ImageTransform flipTransform1 = new FlipImageTransform(rng);
         ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));

         List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, flipTransform2});
         /**
          * Normalization
          **/
         log.info("Fitting to dataset");
         ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);

         MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
             .seed(seed)
             .iterations(iterations)
             .regularization(false).l2(0.005) // tried 0.0001, 0.0005
             .activation(Activation.RELU)
             .learningRate(0.05) // tried 0.001, 0.005, 0.01
             .weightInit(WeightInit.XAVIER)
             .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
             .updater(Updater.ADAM)
             .list()
             .layer(0, convInit("cnn1", channels, 32 ,  new int[]{5, 5}, new int[]{1, 1}, new int[]{0, 0}, 0))
             .layer(1, maxPool("maxpool1", new int[]{2,2}))
             .layer(2, conv3x3("cnn2", 64, 0))
             .layer(3, conv3x3("cnn3", 64,1))
             .layer(4, maxPool("maxpool2", new int[]{2,2}))
             .layer(5, new DenseLayer.Builder().activation(Activation.RELU)
                 .nOut(512).dropOut(0.5).build())
             .layer(6, new OutputLayer.Builder(LossFunctions.LossFunction.RECONSTRUCTION_CROSSENTROPY)
                 .nOut(numLabels)
                 .activation(Activation.SOFTMAX)
                 .build())
             .backprop(true).pretrain(false)
             .setInputType(InputType.convolutional(height, width, channels))
             .build();
         MultiLayerNetwork network = new MultiLayerNetwork(conf);

         network.init();
         // Visualizing Network Training
         BaseImageRecordReader recordReader = new ImageRecordReader(height, width, channels, labelMaker);
         DataSetIterator dataIter;
         MultipleEpochsIterator trainIter;

         log.info("Train model....");
         // Train without transformations
         recordReader.initialize(trainData, null);
         dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
         preProcessor.fit(dataIter);
         dataIter.setPreProcessor(preProcessor);
         trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
         network.fit(trainIter);

         // Train with transformations
         for (ImageTransform transform : transforms) {
             System.out.print("\nTraining on transformation: " + transform.getClass().toString() + "\n\n");
             recordReader.initialize(trainData, transform);
             dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
             preProcessor.fit(dataIter);
             dataIter.setPreProcessor(preProcessor);
             trainIter = new MultipleEpochsIterator(nEpochs, dataIter);
             network.fit(trainIter);
         }

         log.info("Evaluate model....");
         recordReader.initialize(testData);
         dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numLabels);
         preProcessor.fit(dataIter);
         dataIter.setPreProcessor(preProcessor);
         Evaluation eval = network.evaluate(dataIter);
         log.info(eval.stats(true));

         if (save) {
             log.info("Save model....");
             ModelSerializer.writeModel(network,  "bird.bin", true);
         }
         log.info("**************** Retinal Image Classification finished ********************");
     }
    private ConvolutionLayer convInit(String name, int in, int out, int[] kernel, int[] stride, int[] pad, double bias) {
        return new ConvolutionLayer.Builder(kernel, stride, pad).name(name).nIn(in).nOut(out).biasInit(bias).build();
    }
    private ConvolutionLayer conv3x3(String name, int out, double bias) {
        return new ConvolutionLayer.Builder(new int[]{3,3}, new int[] {1,1}, new int[] {1,1}).name(name).nOut(out).biasInit(bias).build();
    }
    private SubsamplingLayer maxPool(String name, int[] kernel) {
        return new SubsamplingLayer.Builder(kernel, new int[]{2,2}).name(name).build();
    }
    private File convert(MultipartFile file) throws IOException {
        File convFile = new File(file.getOriginalFilename());
        convFile.createNewFile();
        FileOutputStream fos = new FileOutputStream(convFile);
        fos.write(file.getBytes());
        fos.close();
        return convFile;
    }

    public void transform(MultipartFile imageFile) {
        NativeImageLoader loader = new NativeImageLoader(40, 90, 4);
        INDArray image = null;
        try {
            image = loader.asMatrix(convert(imageFile));
        } catch (IOException e) {
            e.printStackTrace();
        }
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(image);
        INDArray output = model.output(image, true);
    }
}
