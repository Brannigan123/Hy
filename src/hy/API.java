package hy;

import hy.layer.feedforward.Activation;
import hy.layer.feedforward.AvgPool;
import hy.layer.feedforward.AvgPool.AvgPoolBuilder;
import hy.layer.feedforward.Conv;
import hy.layer.feedforward.Conv.ConvBuilder;
import hy.layer.feedforward.Conv1D;
import hy.layer.feedforward.Conv1D.Conv1DBuilder;
import hy.layer.feedforward.Dropout;
import hy.layer.feedforward.FC;
import hy.layer.feedforward.FC.FCBuilder;
import hy.layer.feedforward.FeedForwardLayer;
import hy.layer.feedforward.Flatten;
import hy.layer.feedforward.MaxPool;
import hy.layer.feedforward.MaxPool.MaxPoolBuilder;
import hy.layer.feedforward.Reshape;
import hy.layer.feedforward.Reshape.ReshapeBuilder;
import hy.layer.feedforward.Scaling;
import hy.layer.feedforward.TConv;
import hy.layer.feedforward.TConv.TConvBuilder;
import hy.layer.feedforward.TConv1D;
import hy.layer.feedforward.TConv1D.TConv1DBuilder;
import hy.loss.Loss;
import hy.model.Model.TrainConfig;
import hy.model.Model.TrainConfig.TrainConfigBuilder;
import hy.model.Sequential;
import hy.optimizer.AdaGrad;
import hy.optimizer.AdaGrad.AdaGradBuilder;
import hy.optimizer.Adam;
import hy.optimizer.Adam.AdamBuilder;
import hy.optimizer.GradOptimizer.Direction;
import hy.optimizer.Momentum;
import hy.optimizer.Momentum.MomentumBuilder;
import hy.optimizer.NAG;
import hy.optimizer.NAG.NAGBuilder;
import hy.optimizer.RMSProp;
import hy.optimizer.RMSProp.RMSPropBuilder;
import hy.optimizer.SGD;
import hy.optimizer.SGD.SGDBuilder;
import hy.regularizer.ElasticNetRegularizer;
import hy.regularizer.L1Regularizer;
import hy.regularizer.L2Regularizer;
import hy.util.NArray;
import hy.util.NArray.NArrayBuilder;
import hy.util.Padding;
import lombok.experimental.UtilityClass;

@UtilityClass
public class API {

    // N-dimensional Array
    public NArrayBuilder NArray() { return NArray.builder(); }

    public NArray NArray(int[] shape, double... data) { return new NArray(shape, data); }

    public NArray NArray(int... shape) { return new NArray(shape); }

    public NArray Array(NArray... data) { return NArray.of(data); }

    public NArray Array(double... data) { return NArray.of(data); }

    public NArray Array(double[]... data) { return NArray.of(data); }

    public NArray Array(double[][]... data) { return NArray.of(data); }

    public NArray Array(double[][][]... data) { return NArray.of(data); }

    // Sequential
    public Sequential Sequential() { return new Sequential(); }

    // Scale
    public Scaling Scale(double factor) { return new Scaling(factor); }

    // Dropout

    public Dropout Dropout() { return new Dropout(0.5); }

    public Dropout Dropout(double rate) { return new Dropout(rate); }

    // Flatten
    public Flatten Flatten() { return new Flatten(); }

    // Reshape

    public ReshapeBuilder Reshape() { return Reshape.builder(); }

    public Reshape Reshape(int... newShape) { return new Reshape(0, newShape); }

    // FC
    public FCBuilder FC() { return FC.builder(); }

    public FC FC(int in, int... out) { return new FC(in, out); }

    // 1D Conv
    public Conv1DBuilder Conv1D() { return Conv1D.builder(); }

    public Conv1D Conv1D(int filterCount, int channels, int winSize, int stride, int padding) {
        return new Conv1D(filterCount, channels, winSize, stride, padding);
    }

    public Conv1D Conv1D(int filterCount, int channels, int winSize, int padding) {
        return new Conv1D(filterCount, channels, winSize, padding);
    }

    public Conv1D Conv1D(int filterCount, int channels, int winSize, int stride, Padding padding) {
        return new Conv1D(filterCount, channels, winSize, stride, padding);
    }

    public Conv1D Conv1D(int filterCount, int channels, int winSize, Padding padding) {
        return new Conv1D(filterCount, channels, winSize, padding);
    }

    public Conv1D Conv1D(int filterCount, int channels, int winSize) {
        return new Conv1D(filterCount, channels, winSize);
    }

    // Conv
    public ConvBuilder Conv() { return Conv.builder(); }

    public Conv Conv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY,
        int paddingX, int paddingY) {
        return new Conv(filterCount, channels, winWidth, winHeight, strideX, strideY, paddingX, paddingY);
    }

    public Conv Conv(int filterCount, int channels, int winSize, int stride, int padding) {
        return new Conv(filterCount, channels, winSize, stride, padding);
    }

    public Conv Conv(int filterCount, int channels, int winSize, int padding) {
        return new Conv(filterCount, channels, winSize, padding);
    }

    public Conv Conv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY,
        Padding padding) {
        return new Conv(filterCount, channels, winWidth, winHeight, strideX, strideY, padding);
    }

    public Conv Conv(int filterCount, int channels, int winSize, int stride, Padding padding) {
        return new Conv(filterCount, channels, winSize, stride, padding);
    }

    public Conv Conv(int filterCount, int channels, int winSize, Padding padding) {
        return new Conv(filterCount, channels, winSize, padding);
    }

    public Conv Conv(int filterCount, int channels, int winSize) {
        return new Conv(filterCount, channels, winSize);
    }

    // Transposed Conv
    public TConvBuilder TConv() { return TConv.builder(); }

    public TConv TConv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY,
        int paddingX, int paddingY) {
        return new TConv(filterCount, channels, winWidth, winHeight, strideX, strideY, paddingX, paddingY);
    }

    public TConv TConv(int filterCount, int channels, int winSize, int stride, int padding) {
        return new TConv(filterCount, channels, winSize, stride, padding);
    }

    public TConv TConv(int filterCount, int channels, int winSize, int padding) {
        return new TConv(filterCount, channels, winSize, padding);
    }

    public TConv TConv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY,
        Padding padding) {
        return new TConv(filterCount, channels, winWidth, winHeight, strideX, strideY, padding);
    }

    public TConv TConv(int filterCount, int channels, int winSize, int stride, Padding padding) {
        return new TConv(filterCount, channels, winSize, stride, padding);
    }

    public TConv TConv(int filterCount, int channels, int winSize, Padding padding) {
        return new TConv(filterCount, channels, winSize, padding);
    }

    public TConv TConv(int filterCount, int channels, int winSize) {
        return new TConv(filterCount, channels, winSize);
    }

    // 1D Transposed Conv
    public TConv1DBuilder TConv1D() { return TConv1D.builder(); }

    public TConv1D TConv1D(int filterCount, int channels, int winSize, int stride, int padding) {
        return new TConv1D(filterCount, channels, winSize, stride, padding);
    }

    public TConv1D TConv1D(int filterCount, int channels, int winSize, int padding) {
        return new TConv1D(filterCount, channels, winSize, padding);
    }

    public TConv1D TConv1D(int filterCount, int channels, int winSize, int stride, Padding padding) {
        return new TConv1D(filterCount, channels, winSize, stride, padding);
    }

    public TConv1D TConv1D(int filterCount, int channels, int winSize, Padding padding) {
        return new TConv1D(filterCount, channels, winSize, padding);
    }

    public TConv1D TConv1D(int filterCount, int channels, int winSize) {
        return new TConv1D(filterCount, channels, winSize);
    }

    // MaxPool
    public MaxPoolBuilder MaxPool() { return MaxPool.builder(); }

    public MaxPool MaxPool(int winWidth, int winHeight, int strideX, int strideY) {
        return new MaxPool(winWidth, winHeight, strideX, strideY);
    }

    public MaxPool MaxPool(int winSize, int stride) { return new MaxPool(winSize, stride); }

    public MaxPool MaxPool(int winSize) { return new MaxPool(winSize); }

    // AvgPool
    public AvgPoolBuilder AvgPool() { return AvgPool.builder(); }

    public AvgPool AvgPool(int winWidth, int winHeight, int strideX, int strideY) {
        return new AvgPool(winWidth, winHeight, strideX, strideY);
    }

    public AvgPool AvgPool(int winSize, int stride) { return new AvgPool(winSize, stride); }

    public AvgPool AvgPool(int winSize) { return new AvgPool(winSize); }

    // Activation
    public final FeedForwardLayer Linear      = Activation.linear;
    public final FeedForwardLayer Sigmoid     = Activation.sigmoid;
    public final FeedForwardLayer HardSigmoid = Activation.hardSigmoid;
    public final FeedForwardLayer Tanh        = Activation.tanh;
    public final FeedForwardLayer Gauss       = Activation.gauss;
    public final FeedForwardLayer ReLU        = Activation.relu;
    public final FeedForwardLayer ReLU6       = Activation.relu6;
    public final FeedForwardLayer ReLU3       = Activation.relu3;
    public final FeedForwardLayer LRelU       = Activation.lrelu;
    public final FeedForwardLayer ElU         = Activation.elu;
    public final FeedForwardLayer SELU        = Activation.selu;
    public final FeedForwardLayer Softmax     = Activation.softmax;

    public FeedForwardLayer lrelu(double alpha) { return Activation.lrelu(alpha); }

    public FeedForwardLayer elu(double alpha) { return Activation.elu(alpha); }

    public FeedForwardLayer radial(double altitude) { return Activation.radial(altitude); }

    public FeedForwardLayer softplus(double beta) { return Activation.softplus(beta); }

    // train config
    public TrainConfigBuilder TrainConfig() { return TrainConfig.builder(); }

    // Loss
    public final Loss    SquareLoss          = Loss.square;
    public final Loss    LogCoshLoss         = Loss.logCosh;
    public final Loss    SoftmaxCrossEntropy = Loss.softmaxCrossEntropy;
    public final Loss    BinaryCrossEntropy  = Loss.binaryCrossEntropy;

    // L1 Regularizer
    public L1Regularizer L1Regularizer       = hy.regularizer.L1Regularizer.Default;

    public L1Regularizer L1Regularizer(double lambda) { return hy.regularizer.L1Regularizer.lambda(lambda); }

    // L2 Regularizer
    public final L2Regularizer L2Regularizer = hy.regularizer.L2Regularizer.Default;

    public L2Regularizer L2Regularizer(double lambda) { return hy.regularizer.L2Regularizer.lambda(lambda); }

    // L2 Regularizer
    public final ElasticNetRegularizer ElasticNetRegularizer = hy.regularizer.ElasticNetRegularizer.Default;

    public ElasticNetRegularizer ElasticNetRegularizer(double l1_lambda, double l2_lambda) {
        return hy.regularizer.ElasticNetRegularizer.lambda(l1_lambda, l2_lambda);
    }

    // SGD
    public SGDBuilder<?, ?> SGD() { return SGD.builder(); }

    public SGD SGD(double learningRate, Direction direction) { return new SGD(learningRate, direction); }

    public SGD SGD(double learningRate) { return new SGD(learningRate); }

    public SGD SGD(Direction direction) { return new SGD(direction); }

    // Momentum
    public MomentumBuilder<?, ?> Momentum() { return Momentum.builder(); }

    public Momentum Momentum(double learningRate, Direction direction, double mu) {
        return new Momentum(learningRate, direction, mu);
    }

    public Momentum Momentum(double learningRate, Direction direction) {
        return new Momentum(learningRate, direction);
    }

    public Momentum Momentum(double learningRate, double mu) { return new Momentum(learningRate, mu); }

    public Momentum Momentum(double learningRate) { return new Momentum(learningRate); }

    public Momentum Momentum(Direction direction) { return new Momentum(direction); }

    // NAG
    public NAGBuilder<?, ?> NAG() { return NAG.builder(); }

    public NAG NAG(double learningRate, Direction direction, double mu) {
        return new NAG(learningRate, direction, mu);
    }

    public NAG NAG(double learningRate, Direction direction) { return new NAG(learningRate, direction); }

    public NAG NAG(double learningRate, double mu) { return new NAG(learningRate, mu); }

    public NAG NAG(double learningRate) { return new NAG(learningRate); }

    public NAG NAG(Direction direction) { return new NAG(direction); }

    // RMSProp
    public RMSPropBuilder<?, ?> RMSProp() { return RMSProp.builder(); }

    public RMSProp RMSProp(double learningRate, Direction direction, double mu) {
        return new RMSProp(learningRate, direction, mu);
    }

    public RMSProp RMSProp(double learningRate, Direction direction) {
        return new RMSProp(learningRate, direction);
    }

    public RMSProp RMSProp(double learningRate, double mu) { return new RMSProp(learningRate, mu); }

    public RMSProp RMSProp(double learningRate) { return new RMSProp(learningRate); }

    public RMSProp RMSProp(Direction direction) { return new RMSProp(direction); }

    // AdaGrad
    public AdaGradBuilder<?, ?> AdaGrad() { return AdaGrad.builder(); }

    public RMSProp AdaGrad(double learningRate, Direction direction, double mu) {
        return new RMSProp(learningRate, direction, mu);
    }

    public AdaGrad AdaGrad(double learningRate, Direction direction) {
        return new AdaGrad(learningRate, direction);
    }

    public AdaGrad AdaGrad(double learningRate) { return new AdaGrad(learningRate); }

    public AdaGrad AdaGrad(Direction direction) { return new AdaGrad(direction); }

    // Adam
    public AdamBuilder<?, ?> Adam() { return Adam.builder(); }

    public Adam Adam(double learningRate, Direction direction, double beta1, double beta2) {
        return new Adam(learningRate, direction, beta1, beta2);
    }

    public Adam Adam(double learningRate, Direction direction) { return new Adam(learningRate, direction); }

    public Adam Adam(double learningRate, double beta1, double beta2) {
        return new Adam(learningRate, beta1, beta2);
    }

    public Adam Adam(double learningRate) { return new Adam(learningRate); }

    public Adam Adam(Direction direction) { return new Adam(direction); }
}
