package io.github.jhipster.sample.prediction;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.opencv_core;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import javax.annotation.PostConstruct;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgproc.INTER_AREA;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

@Service
public class RetinalPictureUtils {
    @Autowired
    ResourceLoader resourceLoader;

    MultiLayerNetwork model;

    public void init() throws IOException {
        Resource resourceModel = resourceLoader.getResource("classpath:bird.bin");
        File savedModel = resourceModel.getFile();
        model = ModelSerializer.restoreMultiLayerNetwork(savedModel) ;
    }

    public Boolean retinalclassifier(MultipartFile imageFile) throws IOException{
        NativeImageLoader loader = new NativeImageLoader(32, 32, 3);
        INDArray image = loader.asMatrix(convert(imageFile));
        ImagePreProcessingScaler preProcessor = new ImagePreProcessingScaler(0, 1);
        preProcessor.transform(image);
        INDArray output = model.output(image, false);
        return output.getFloat(0) > 0.8;
    }

    public void resizeRetinaImage(MultipartFile retina) {
        opencv_core.Mat src = imread((BytePointer) retina);
        opencv_core.Mat resizeimage = new opencv_core.Mat();
        opencv_core.Size scale = new opencv_core.Size(400,400);
        resize(src, resizeimage, scale, 0, 0, INTER_AREA);
    }

    private File convert(MultipartFile file) throws IOException {
        File convFile = new File(file.getOriginalFilename());
        convFile.createNewFile();
        FileOutputStream fos = new FileOutputStream(convFile);
        fos.write(file.getBytes());
        fos.close();
        return convFile;
    }
}
