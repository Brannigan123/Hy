package hy.test.gan;

import static hy.API.*;
import static io.vavr.API.printf;
import static io.vavr.API.println;
import static java.util.function.Predicate.not;

import java.util.ArrayList;
import java.util.List;

import java.util.function.Supplier;

import hy.model.Model.TrainConfig.TrainConfigBuilder;
import hy.model.Sequential;
import hy.util.Image;
import hy.util.NArray;
import io.vavr.collection.Array;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Gan {

    String             datapath                = "C:\\Users\\Pres\\Desktop\\ME\\.hub\\face_segment_data_sampleset\\images\\real\\";
    String             genepath                = "C:\\Users\\Pres\\Desktop\\ME\\gen\\gan\\";

    int                randomCount             = 40;
    Supplier<NArray>   randomInput             = () -> NArray(randomCount).randomize();

    Sequential         encoder                 = Sequential()                                                                      // 256
            .add(Conv(4, 3, 3))                                                                                                    // 254
            .add(Conv(5, 4, 3))                                                                                                    // 252
            .add(MaxPool(2), Gauss)                                                                                                // 126
            .add(Conv(6, 5, 3))                                                                                                    // 124
            .add(Conv(6, 6, 3))                                                                                                    // 122
            .add(MaxPool(2), Gauss)                                                                                                // 61
            .add(Conv(5, 6, 3))                                                                                                    // 59
            .add(Conv(5, 5, 3))                                                                                                    // 57
            .add(MaxPool(3), Gauss)                                                                                                // 19
            .flatten()                                                                                                             //
            .add(FC(1805, 1000), Tanh)                                                                                             //
            .dropout(0.3)                                                                                                          //
            .add(FC(1000, 500), Tanh)                                                                                              //
            .dropout(0.2)                                                                                                          //
            .add(FC(500, 50), Tanh)                                                                                                //
            .load("3_256_256 to 50 encoder.enc")                                                                                   //
    ;

    Sequential         decoder                 = Sequential()                                                                      //
            .add(FC(50, 500), Tanh)                                                                                                //
            .dropout(0.3)                                                                                                          //
            .add(FC(500, 1000), Tanh)                                                                                              //
            .dropout(0.2)                                                                                                          //
            .add(FC(1000, 2000), Tanh)                                                                                             //
            .reshape(5, 20, 20)                                                                                                    //
            .add(TConv(6, 5, 3), Gauss)                                                                                            // 22
            .add(TConv(5, 6, 3), Gauss)                                                                                            // 24
            .add(TConv(4, 5, 3), Gauss)                                                                                            // 26
            .add(TConv(3, 4, 3), Gauss)                                                                                            // 28
            .add(TConv(3, 3, 5, 3, 0), Gauss)                                                                                      // 86
            .add(TConv(3, 3, 5, 3, 2))                                                                                             // 256
            .load("50 to 3_256_256 decoder.dec")                                                                                   //
    ;

    Sequential         distributionTransformer = Sequential()                                                                      //
            .add(FC(randomCount, 60), Tanh)                                                                                        //
            .add(FC(60, 70), Tanh)                                                                                                 //
            .add(FC(70, 50), Tanh)                                                                                                 //
            .load("dist tranformer.tran")                                                                                          //
    ;

    Sequential         discriminator           = Sequential()                                                                      //
            .add(FC(50, 20), Tanh)                                                                                                 //
            .add(FC(20, 10), Tanh)                                                                                                 //
            .add(FC(10, 1))                                                                                                        //
            .load("50 encoded discrminator.dis")                                                                                   //
    ;

    Sequential         scammer                 = Sequential()                                                                      //
            .add(distributionTransformer)                                                                                          //
            .dropout(0.2)                                                                                                          //
            .add(decoder)                                                                                                          //
    ;

    Sequential         police                  = Sequential()                                                                      //
            .add(encoder)                                                                                                          //
            .dropout(0.2)                                                                                                          //
            .add(discriminator)                                                                                                    //
    ;

    Sequential         system                  = Sequential()                                                                      //
            .add(scammer)                                                                                                          //
            .dropout(0.2)                                                                                                          //
            .add(police)                                                                                                           //
    ;

    TrainConfigBuilder policeConfig            = TrainConfig()                                                                     //
            .batchSize(15)                                                                                                         //
            .shuffle(true)                                                                                                         //
            .epochs(10)                                                                                                            //
            .regularizer(L2Regularizer(0.006))                                                                                     //
            .freezeWhere(encoder::contains);

    TrainConfigBuilder systemConfig            = TrainConfig()                                                                     //
            .batchSize(15)                                                                                                         //
            .shuffle(true)                                                                                                         //
            .epochs(5)                                                                                                             //
            .regularizer(L2Regularizer(0.061))                                                                                     //
            .freezeWhere(not(distributionTransformer::contains))                                                                   //
    ;

    public void main(String[] args) {
        println(system);
        val episodes = 1000;
        for (var i = 1; i <= episodes; i++) {
            printf("Episode %d\n\n", i);
            trainEpisode();
            println(system.paramStatistics());
            println("\n");
        }
    }

    void trainEpisode() {
        println("discriminator");
        val samples = trainPolice();
        discriminator.save("50 encoded discrminator.dis");
        println("saved discriminator");

        println("transformer");
        trainScammer(samples);
        distributionTransformer.save("dist tranformer.tran");
        println("saved transformer");
    }

    int trainPolice() {
        val x = new ArrayList<NArray>();
        val y = new ArrayList<NArray>();
        insertRealData(x, y);
        insertSyntheticData(x.size(), x, y);
        val config = policeConfig//
                .clearInputs().inputs(x)//
                .clearTargets().targets(y)//
                .build();
        police.fit(config);
        return x.size();
    }

    void trainScammer(int count) {
        val x = new ArrayList<NArray>();
        val y = new ArrayList<NArray>();
        val label = Array(1.0);
        for (var i = 0; i < count; i++) {
            x.add(randomInput.get());
            y.add(label);
        }
        val config = systemConfig//
                .clearInputs().inputs(x)//
                .clearTargets().targets(y)//
                .build();
        system.fit(config);
    }

    void insertRealData(List<NArray> x, List<NArray> y) {
        val label = Array(1.0);
        Array.of(Image.paths(datapath)).shuffle().take(10).asJava().forEach(imagPath -> {
            x.add(Image.read(imagPath, 256, 256));
            y.add(label);
        });
    }

    void insertSyntheticData(int count, List<NArray> x, List<NArray> y) {
        val label = Array(0.0);
        for (var i = 0; i < count; i++) {
            val xi = scammer.predict(randomInput.get());
            x.add(xi);
            y.add(label);
            Image.write(xi, 256, 256, genepath + System.currentTimeMillis() + ".png");
        }
    }

}
