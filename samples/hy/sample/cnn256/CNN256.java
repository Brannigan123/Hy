package hy.sample.cnn256;

import static hy.API.Conv;
import static hy.API.FC;
import static hy.API.Gauss;
import static hy.API.L2Regularizer;
import static hy.API.MaxPool;
import static hy.API.NArray;
import static hy.API.Sequential;
import static hy.API.Softmax;
import static hy.API.SoftmaxCrossEntropy;
import static hy.API.Tanh;
import static hy.API.TrainConfig;
import static io.vavr.API.Array;
import static io.vavr.API.printf;
import static io.vavr.API.println;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import hy.model.Model.EpochLossCallback;
import hy.model.Sequential;
import hy.util.Image;
import hy.util.NArray;
import io.vavr.Tuple2;
import io.vavr.collection.Array;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class CNN256 {
    File       datapath         = new File("samples/data/face_sample_dataset/images/");
    File[]     categories       = datapath.listFiles();

    String     featureParamPath = "cnn feature extractor.pm";
    String     fcParamPath      = "cnn fc.pm";

    Sequential featureExtractor = Sequential()                                         // 3*256*256
            .add(Conv(4, 3, 3))                                                        // 4*254*254
            .add(Conv(5, 4, 3))                                                        // 5*252*252
            .add(MaxPool(2), Gauss)                                                    // 5*126*126
            .dropout(0.3)                                                              // 5*126*126
            .add(Conv(6, 5, 3))                                                        // 6*124*124
            .add(Conv(6, 6, 3))                                                        // 6*122*122
            .add(MaxPool(2), Gauss)                                                    // 6*61*61
            .dropout(0.3)                                                              // 6*61*61
            .add(Conv(5, 6, 3))                                                        // 5*59*59
            .add(Conv(5, 5, 3))                                                        // 5*57*57
            .add(MaxPool(3), Gauss)                                                    // 5*19*19
            .load(featureParamPath)                                                    //
    ;

    Sequential classifier       = Sequential()                                         // 5*19*19
            .flatten()                                                                 // 1805
            .add(FC(1805, 1000), Tanh)                                                 // 1000
            .dropout(0.3)                                                              // 1000
            .add(FC(1000, 500), Tanh)                                                  // 500
            .dropout(0.2)                                                              // 500
            .add(FC(500, 50), Tanh)                                                    // 50
            .add(FC(50, 20), Tanh)                                                     // 20
            .add(FC(20, categories.length), Softmax)                                   // # classes
            .load(fcParamPath)                                                         //
    ;

    Sequential cnn              = Sequential()                                         // 3*256*256
            .add(featureExtractor)                                                     // 5*19*19
            .add(classifier)                                                           // # classes
    ;

    public void main(String[] args) {
        val x = new ArrayList<NArray>();
        val y = new ArrayList<NArray>();
        val xTest = new ArrayList<NArray>();
        val yTest = new ArrayList<NArray>();
        loadData(x, y, xTest, yTest);

        val trainConfig = TrainConfig() //
                .inputs(x)//
                .targets(y) //
                .batchSize(20) //
                .shuffle(true) //
                .loss(SoftmaxCrossEntropy) //
                .epochLossCallBack(CNN256::epochCallback) //
                .minLoss(1e-3) //
                .regularizer(L2Regularizer(0.006)) //
                .epochs(50) //
                .build();

        cnn.fit(trainConfig);

        printf("%.2f%% training accuracy\n", accuracy(x, y));
        printf("%.2f%% test accuracy\n", accuracy(xTest, yTest));
    }

    void loadData(List<NArray> x, List<NArray> y, List<NArray> xTest, List<NArray> yTest) {
        for (int i = 0; i < categories.length; i++) {
            val catpath = categories[i].getAbsolutePath();
            val label = NArray(categories.length)[i] = 1;
            val paths = Array(Image.paths(catpath));
            val sets = paths.splitAt(paths.size() / 2);
            readData(sets._1, label, x, y);
            readData(sets._2, label, xTest, yTest);
        }
    }

    @SuppressWarnings("deprecation")
    void readData(Array<String> paths, NArray label, List<NArray> x, List<NArray> y) {
        paths.forEach(imagPath -> {
            x.add(Image.read(imagPath, 256, 256));
            y.add(label);
        });
    }

    void epochCallback(long epoch, long totalEpochs, String lossType, double loss) {
        println(EpochLossCallback.format(epoch, totalEpochs, lossType, loss));
        println(cnn.paramStatistics());
        if (epoch % 2 == 0) {
            println("Saving current parameters");
            featureExtractor.save(featureParamPath);
            classifier.save(fcParamPath);
            println("Saved");
        }
    }

    double accuracy(List<NArray> x, List<NArray> y) {
        val modCount = new AtomicInteger();
        val accuracy = Array.ofAll(x).zip(y)//
                .map(instance -> {
                    val a = labelFrom(instance._2);
                    val p = labelFrom(cnn.predict(instance._1));
                    printf("\r%-3s", ".".repeat(modCount.getAndUpdate(mod -> (++mod) % 4)));
                    // println("label : " + a + ", prediction : " + p);
                    return p.equals(a) ? 1 : 0;
                }).average().getOrElse(0.0) * 100.0;
        printf("\r");
        return accuracy;
    }

    String labelFrom(NArray logits) {
        return Array.ofAll(logits).zipWithIndex()//
                .<Double>maxBy(Tuple2::_1)//
                .map(pi -> categories[pi._2].getName())//
                .get();
    }
}
