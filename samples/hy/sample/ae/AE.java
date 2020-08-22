package hy.sample.ae;

import static hy.API.*;
import static io.vavr.API.Array;
import static io.vavr.API.printf;
import static io.vavr.API.println;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import hy.model.Sequential;
import hy.model.Model.EpochLossCallback;
import hy.util.Image;
import hy.util.NArray;
import io.vavr.collection.Array;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class AE {

    File       datapath         = new File("samples/data/face_sample_dataset/images/");
    File[]     categories       = datapath.listFiles();

    String     dumpPath         = "samples/dump/ae";
    String     encoderParamPath = "ae encoder.pm";
    String     decoderParamPath = "ae decoder.pm";

    Sequential encoder          = Sequential()                                         // 3*256*256
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
            .load(encoderParamPath)                                                    //
    ;

    Sequential decoder          = Sequential()                                         // 5*19*19
            .add(TConv(5, 5, 3, 3, 0))                                                 // 5*57*57
            .add(TConv(5, 5, 3), Gauss)                                                // 5*59*59
            .dropout(0.3)                                                              // 5*59*59
            .add(TConv(6, 5, 3))                                                       // 6*61*61
            .add(TConv(6, 6, 4, 2, 0), Gauss)                                          // 6*124*124
            .dropout(0.3)                                                              // 6*124*124
            .add(TConv(5, 6, 3))                                                       // 5*126*126
            .add(TConv(4, 5, 3, 2, 0), Gauss)                                          // 5*253*253
            .add(TConv(3, 4, 4, 1, 0))                                                 // 3*256*256
            .load(decoderParamPath)                                                    //
    ;

    Sequential ae               = Sequential()                                         //
            .add(encoder)                                                              //
            .add(decoder)                                                              //
    ;

    public void main(String[] args) {
        val x = new ArrayList<NArray>();
        val xTest = new ArrayList<NArray>();
        loadData(x, xTest);

        val trainConfig = TrainConfig() //
                .inputs(x)//
                .targets(x) //
                .batchSize(20) //
                .shuffle(true) //
                .loss(LogCoshLoss)//
                .epochLossCallBack(AE::epochCallback) //
                .minLoss(1e-3) //
                // .regularizer(L2Regularizer(0.006)) //
                .epochs(500) //
                .build();

        ae.fit(trainConfig);

        printf("%.2f%% training accuracy\n", accuracy(x, "training"));
        printf("%.2f%% test accuracy\n", accuracy(xTest, "test"));
    }

    @SuppressWarnings("deprecation")
    void loadData(List<NArray> x, List<NArray> xTest) {
        for (val category : categories) {
            val catpath = category.getAbsolutePath();
            val paths = Array(Image.paths(catpath)).shuffle().take(4);
            val sets = paths.splitAt(paths.size() / 2);
            sets._1.forEach(imagPath -> x.add(Image.read(imagPath, 256, 256)));
            sets._2.forEach(imagPath -> xTest.add(Image.read(imagPath, 256, 256)));
        }
    }

    void epochCallback(long epoch, long totalEpochs, String lossType, double loss) {
        println(EpochLossCallback.format(epoch, totalEpochs, lossType, loss));
        // println(ae.paramStatistics());
        if (epoch % 2 == 0) {
            println("Saving current parameters");
            encoder.save(encoderParamPath);
            decoder.save(decoderParamPath);
            println("Saved");
        }
    }

    double accuracy(List<NArray> x, String prefix) {
        val index = new AtomicInteger();
        val accuracy = Array.ofAll(x)//
                .map(instance -> {
                    val a = instance;
                    val p = ae.predict(instance);
                    Image.write(a, 256, 256, dumpPath + "/" + prefix + index.get() + ".png");
                    Image.write(p, 256, 256, dumpPath + "/" + prefix + index.get() + " rec.png");
                    printf("\r%-3s", ".".repeat(index.incrementAndGet() % 4));
                    return 1.0 - (p - a).map(Math::abs).reduce(0, Double::sum) / p.size;
                }).average().getOrElse(0.0) * 100.0;
        printf("\r");
        return accuracy;
    }
}
