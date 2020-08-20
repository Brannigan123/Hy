package hy.model;

import static io.vavr.API.printf;
import static io.vavr.API.println;

import java.util.List;
import java.util.Set;
import java.util.function.Predicate;

import hy.layer.ParamLayer;
import hy.loss.Loss;
import hy.optimizer.Adam;
import hy.optimizer.GradOptimizer.Direction;
import hy.optimizer.Optimizer;
import hy.regularizer.Regularizer;
import hy.util.Format;
import hy.util.NArray;
import io.vavr.CheckedRunnable;
import io.vavr.Tuple2;
import io.vavr.collection.Stream;
import io.vavr.control.Try;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.NonNull;
import lombok.Singular;
import lombok.experimental.FieldDefaults;

public interface Model {

    public List<NArray> predict(Iterable<? extends NArray> inputs);

    public NArray[] predict(NArray... inputs);

    public NArray predict(NArray input);

    public Model fit(TrainConfig trainConfig);

    public Model save(String path);

    public Model load(String path);

    @Builder
    @FieldDefaults(makeFinal = true)
    public static class TrainConfig {
        @Default boolean                                              shuffle                   = false;
        @Default int                                                  batchSize                 = 1;
        @Default long                                                 epochs                    = 100;
        @Default double                                               minLoss                   = Double.MIN_VALUE;
        @Default double                                               maxLoss                   = Double.POSITIVE_INFINITY;
        @NonNull @Default Loss                                        loss                      = Loss.square;
        @NonNull @Default Optimizer                                   optimizer                 = new Adam(
                Direction.DESCEND);
        Regularizer                                                   regularizer;
        @NonNull @Default Predicate<ParamLayer>                       freezeWhere               = layer -> false;
        @NonNull @Singular(ignoreNullCollections = true) Set<Long>    freezeLayers;
        @NonNull @Singular(ignoreNullCollections = true) List<NArray> inputs;
        @NonNull @Singular(ignoreNullCollections = true) List<NArray> targets;
        @NonNull @Default InputOutputTargetCallback                   inputOutputTargetCallback = (input, output,
            target) -> {};
        @NonNull @Default BatchLossCallback                           batchLossCallback         = BatchLossCallback.Default;
        @NonNull @Default EpochLossCallback                           epochLossCallBack         = EpochLossCallback.Default;

        public final Stream<Tuple2<NArray, NArray>> trainData() { return Stream.ofAll(inputs).zip(targets); }

        public final boolean isNotFrozen(long index, ParamLayer layer) {
            return !(freezeLayers.contains(index) || freezeWhere.test(layer));
        }

    }

    public static interface InputOutputTargetCallback {
        public void accept(NArray input, NArray output, NArray target);
    }

    public static interface BatchLossCallback {
        public void accept(long batch, int procesedBatchItems, int totalBatchItems, String lossType, double loss,
            double averageLoss);

        public static String format(int barLength, long batch, int procesedBatchItems, int totalBatchItems,
            String lossType, double loss, double averageLoss) {
            return String.format("\r%s Average %s = %s, %s of Batch %d",
                Format.fmtFractionBar(barLength, procesedBatchItems, totalBatchItems), lossType,
                Format.fmtDecimal(averageLoss), Format.fmtFraction(procesedBatchItems, totalBatchItems), batch);
        }

        public static final BatchLossCallback Default = new BatchLossCallback() {
            @Override
            public void accept(long batch, int procesedBatchItems, int totalBatchItems, String lossType, double loss,
                double averageLoss) {
                printf(format(50, batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss));
            }
        };

        public default BatchLossCallback then(@NonNull BatchLossCallback after) {
            return (batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss) -> {
                accept(batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss);
                after.accept(batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss);
            };
        }

        public default BatchLossCallback then(@NonNull CheckedRunnable after) {
            return (batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss) -> {
                accept(batch, procesedBatchItems, totalBatchItems, lossType, loss, averageLoss);
                Try.run(after);
            };
        }
    }

    public static interface EpochLossCallback {

        public void accept(long epoch, long totalEpochs, String lossType, double loss);

        public static String format(long epoch, long totalEpochs, String lossType, double loss) {
            return String.format("\repoch %s %s = %s", Format.fmtFraction(epoch, totalEpochs), lossType,
                Format.fmtDecimal(loss));
        }

        public static final EpochLossCallback Default = new EpochLossCallback() {
            @Override
            public void accept(long epoch, long totalEpochs, String lossType, double loss) {
                println(format(epoch, totalEpochs, lossType, loss));
            }

        };

        public default EpochLossCallback then(@NonNull EpochLossCallback after) {
            return (epoch, totalEpochs, lossType, loss) -> {
                accept(epoch, totalEpochs, lossType, loss);
                after.accept(epoch, totalEpochs, lossType, loss);
            };
        }

        public default EpochLossCallback then(@NonNull CheckedRunnable after) {
            return (epoch, totalEpochs, lossType, loss) -> {
                accept(epoch, totalEpochs, lossType, loss);
                Try.run(after);
            };
        }
    }

}
