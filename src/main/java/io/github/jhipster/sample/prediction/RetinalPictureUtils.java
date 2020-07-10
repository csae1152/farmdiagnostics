package io.github.jhipster.sample.prediction;

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

@Service
public class RetinalPictureUtils {
    @Autowired
    ResourceLoader resourceLoader;

    MultiLayerNetwork model;

    @PostConstruct
    private void init() throws IOException {
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

    private File convert(MultipartFile file) throws IOException {
        File convFile = new File(file.getOriginalFilename());
        convFile.createNewFile();
        FileOutputStream fos = new FileOutputStream(convFile);
        fos.write(file.getBytes());
        fos.close();
        return convFile;
    }
}
