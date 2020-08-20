package hy.model;

import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import hy.layer.Layer;
import hy.layer.ParamLayer;
import hy.layer.ParamStatistics;
import hy.layer.feedforward.Dropout;
import hy.layer.feedforward.Flatten;
import hy.layer.feedforward.Reshape;
import hy.layer.feedforward.Scaling;
import hy.util.NArray;
import hy.util.StreamUtil;
import lombok.NonNull;
import lombok.Synchronized;
import lombok.val;
import lombok.var;
import lombok.experimental.ExtensionMethod;
import lombok.extern.java.Log;

@Log
@ExtensionMethod({ StreamUtil.class })
public class Sequential implements Model, Iterable<Layer> {

    private LinkedList<Layer> layers = new LinkedList<>();

    public Layer get(int index) { return layers.get(index); }

    @Synchronized("layers")
    public Sequential add(@NonNull Layer layer) {
        layers.add(layer);
        return this;
    }

    @Synchronized("layers")
    public Sequential add(@NonNull Layer... layers) {
        this.layers.addAll(List.of(layers));
        return this;
    }

    @Synchronized("layers")
    public Sequential add(@NonNull Iterable<? extends Layer> layers) {
        layers.forEach(this.layers::add);
        return this;
    }

    @Synchronized("layers")
    public Sequential dropout(double rate) {
        this.layers.add(new Dropout(rate));
        return this;
    }

    @Synchronized("layers")
    public Sequential scale(double factor) {
        this.layers.add(new Scaling(factor));
        return this;
    }

    @Synchronized("layers")
    public Sequential flatten() {
        layers.add(new Flatten());
        return this;
    }

    @Synchronized("layers")
    public Sequential reshape(@NonNull int... newShape) {
        layers.add(new Reshape(0, newShape));
        return this;
    }

    @Override
    public List<NArray> predict(@NonNull Iterable<? extends NArray> inputs) {
        val predictions = new LinkedList<NArray>();
        for (val input : inputs) predictions.add(Objects.requireNonNull(predict(input)));
        return Collections.unmodifiableList(predictions);
    }

    @Override
    public NArray[] predict(@NonNull NArray... inputs) {
        val n = inputs.length;
        val predictions = new NArray[n];
        for (int i = 0; i < n; i++) predictions[i] = predict(Objects.requireNonNull(inputs[i]));
        return predictions;
    }

    @Override
    public NArray predict(@NonNull NArray input) {
        var prediction = input;
        for (val layer : layers) prediction = layer.of(prediction, false);
        return prediction;
    }

    @Override
    public Sequential fit(@NonNull TrainConfig config) {
        val lossType = config.loss.toString();
        val epochs = config.epochs;
        val minLoss = config.minLoss, maxLoss = config.maxLoss;
        for (var epoch = 0L; epoch < epochs;) {
            val loss = fitEpoch(config);
            config.epochLossCallBack.accept(++epoch, epochs, lossType, loss);
            if (loss <= minLoss || loss >= maxLoss) break;
        }
        return this;
    }

    public double fitEpoch(@NonNull TrainConfig config) {
        val layers = this.layers.stream().toList();
        val pLayers = layers.stream().filter(ParamLayer.class).toList();
        val data = config.shuffle ? config.trainData().shuffle() : config.trainData();
        val lossFn = config.loss, lossType = lossFn.toString();
        val batchSize = config.batchSize;
        var totalLoss = 0.0, i = 0L, bi = 0;
        for (val instance : data) {
            val input = instance._1, target = instance._2;
            val results = trainingPrediction(layers, input);
            val prediction = results.getLast();
            val loss = lossFn.of(prediction, target).reduce(0, Double::sum);

            totalLoss += loss;
            backprop(layers, results, lossFn.gradient(prediction, target));

            if ((bi = (int) (++i % batchSize)) == 0) update(config, pLayers);
            val b = (long) Math.ceil((double) i / (double) batchSize);

            config.batchLossCallback.accept(b, bi == 0 ? batchSize : bi, batchSize, lossType, loss, totalLoss / i);
        }
        if (bi != 0) update(config, pLayers);
        return totalLoss / i;
    }

    private LinkedList<NArray> trainingPrediction(List<Layer> layers, @NonNull NArray input) {
        val results = new LinkedList<NArray>();
        var result = input;
        results.add(result);
        for (val layer : layers) results.add(result = layer.of(result, true));
        return results;
    }

    private void backprop(LinkedList<Layer> layers, LinkedList<NArray> results, NArray loss) {
        val layerIt = layers.descendingIterator(), resultIt = results.descendingIterator();
        var output = resultIt.next();
        while (layerIt.hasNext()) {
            val layer = layerIt.next();
            val input = resultIt.next();
            loss = layer.delta(input, output, loss);
            output = input;
        }
    }

    @Synchronized("layers")
    private void update(TrainConfig config, List<ParamLayer> layers) {
        val optimizer = config.optimizer, regularizer = config.regularizer;
        var i = 0L;
        for (val layer : layers) {
            if (config.isNotFrozen(i, layer)) layer.update(optimizer, regularizer);
            else layer.clearGradients();
            i++;
        }
        optimizer.advance();
    }

    @Override
    @Synchronized("layers")
    public Sequential save(@NonNull String path) {
        try {
            val pLayers = layers.stream().filter(ParamLayer.class).toList();
            val size = pLayers.stream().mapToInt(ParamLayer::bytes).sum();
            val buf = ByteBuffer.allocate(size), dbuf = buf.asDoubleBuffer();
            pLayers.forEach(layer -> dbuf.put(layer.parameterBuffer()));
            Files.write(Paths.get(path), buf.flip().array());
            log.fine(String.format("Sucessfully saved model to '%s'", path));
        } catch (Exception e) {
            log.warning(String.format("Failed to save model to '%s' : %s", path, e.toString()));
        }
        return this;
    }

    @Override
    @Synchronized("layers")
    public Sequential load(@NonNull String path) {
        try {
            val buf = ByteBuffer.wrap(Files.readAllBytes(Paths.get(path))).asDoubleBuffer();
            layers.stream().filter(ParamLayer.class).forEach(layer -> layer.readParameter(buf));
            log.fine(String.format("Sucessfully loaded model from '%s'", path));
        } catch (Exception e) {
            log.warning(String.format("Failed to load model from '%s' : %s", path, e.toString()));
        }
        return this;
    }

    @Override
    @Synchronized("layers")
    public String toString() {
        val sb = new StringBuilder();
        layers.forEach(layer -> {
            sb.append(layer).append('\n');
            if (layer instanceof ParamLayer) {
                sb.append(((ParamLayer) layer).paramStatistics());
                sb.append('\n');
            }
            sb.append('\n');
        });
        return sb.toString();
    }

    @Synchronized("layers")
    public ParamStatistics paramStatistics() {
        return layers.stream().filter(ParamLayer.class)//
                .map(ParamLayer::paramStatistics)//
                .reduce(new ParamStatistics(), ParamStatistics::accept);
    }

    @Synchronized("layers")
    public boolean isEmpty() { return layers.isEmpty(); }

    @Synchronized("layers")
    public boolean contains(Object o) { return layers.contains(o); }

    @Override
    @Synchronized("layers")
    public Iterator<Layer> iterator() { return layers.iterator(); }

    @Synchronized("layers")
    public Layer[] toArray() { return layers.toArray(Layer[]::new); }

    @Synchronized("layers")
    public <T> T[] toArray(T[] a) { return layers.toArray(a); }

    @Synchronized("layers")
    public Sequential remove(int index) {
        layers.remove(index);
        return this;
    }

    @Synchronized("layers")
    public Sequential remove(Layer o) {
        layers.remove(o);
        return this;
    }

    @Synchronized("layers")
    public Sequential clear() {
        layers.clear();
        return this;
    }

}
