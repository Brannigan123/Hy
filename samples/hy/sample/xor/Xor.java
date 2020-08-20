package hy.sample.xor;

import static hy.API.*;
import static io.vavr.API.println;

import java.util.List;

import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Xor {

    public void main(String[] args) {
        val nn = Sequential() //
                .add(FC(2, 3), Tanh) //
                .add(FC(3, 2), Tanh) //
                .add(FC(2, 1));

        val x = List.of( //
            Array(0, 0), //
            Array(0, 1), //
            Array(1, 0), //
            Array(1, 1) //
        );

        val y = List.of(Array(0), Array(1), Array(1), Array(0));

        val trainConfig = TrainConfig() //
                .inputs(x).targets(y) //
                .batchSize(2).shuffle(true) //
                .loss(LogCoshLoss) //
                .minLoss(1e-3) //
                .epochs(2000) //
                .build();

        nn.fit(trainConfig);

        println(nn.predict(Array(0, 0)));
        println(nn.predict(Array(0, 1)));
        println(nn.predict(Array(1, 0)));
        println(nn.predict(Array(1, 1)));

    }

}
