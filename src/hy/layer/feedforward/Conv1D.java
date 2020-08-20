package hy.layer.feedforward;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import hy.optimizer.Optimizer;
import hy.regularizer.Regularizer;
import hy.util.NArray;
import hy.util.Padding;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import lombok.var;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class Conv1D implements ParamFeedForwardLayer {

    int                      filterCount;
    int                      channels;
    int                      winSize;
    int                      stride;
    int                      padding;

    NArray                   W;
    NArray                   b;

    NArray                   dW;
    NArray                   db;

    Map<Optimizer, NArray[]> pW;
    Map<Optimizer, NArray[]> pb;

    AtomicLong               changes;

    @Builder
    public Conv1D(int filterCount, int channels, int winSize, int stride, int padding) {
        this.filterCount = filterCount < 1 ? 1 : filterCount;
        this.channels = channels < 1 ? 1 : channels;
        this.winSize = winSize < 1 ? 1 : winSize;
        this.stride = stride < 1 ? 1 : stride;
        this.padding = padding < 0 ? 0 : padding;
        W = new NArray(filterCount, channels, winSize).randomize();
        b = new NArray(filterCount).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
    }

    public Conv1D(int filterCount, int channels, int winSize, int padding) {
        this(filterCount, channels, winSize, 1, padding);
    }

    public Conv1D(int filterCount, int channels, int winSize, int stride, @NonNull Padding padding) {
        this.filterCount = filterCount < 1 ? 1 : filterCount;
        this.channels = channels < 1 ? 1 : channels;
        this.winSize = winSize < 1 ? 1 : winSize;
        this.stride = stride < 1 ? 1 : stride;
        W = new NArray(filterCount, channels, winSize).randomize();
        b = new NArray(filterCount).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
        switch (padding) {
        case SAME:
            if ((winSize - 1) % 2 != 0)
                throw new IllegalArgumentException("Bad convolution parameters. Filter width should be odd.");
            this.padding = (winSize - 1) / 2;
            break;
        case VALID:
        default:
            this.padding = 0;
        }
    }

    public Conv1D(int filterCount, int channels, int winSize, Padding padding) {
        this(filterCount, channels, winSize, 1, padding);
    }

    public Conv1D(int filterCount, int channels, int winSize) { this(filterCount, channels, winSize, 1, 0); }

    @Override
    public NArray of(@NonNull NArray input, boolean isTraining) {
        val dims = input.dims;
        if (dims < 2) throw new IllegalArgumentException("Input dimensions should be atleast 2. Got " + dims);
        val inShape = input.getShape();
        val outShape = inShape.clone();
        outShape[dims - 2] = filterCount;
        outShape[dims - 1] = convolved(inShape[dims - 1], winSize, stride, padding);
        val out = new NArray(outShape).fill(outCoords -> {
            val k = outCoords[dims - 2];
            val i = outCoords[dims - 1] * stride;
            val inCoords = outCoords.clone();
            var sum = 0.0;
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winSize; rx++) {
                    val x = i - padding + rx;
                    if (x < 0 || x >= inShape[dims - 1]) continue;
                    val kcrx = new int[] { k, c, rx };
                    inCoords[dims - 2] = c;
                    inCoords[dims - 1] = x;
                    sum = sum + input[inCoords] * W[kcrx];
                }
            }
            return sum + b[k];
        });
        return out;
    }

    @Override
    public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
        val dX = NArray.like(input);
        val inShape = input.getShape();
        val dims = inShape.length;
        delta.coordinates().forEach(outCoords -> {
            val k = outCoords[dims - 2];
            val i = outCoords[dims - 1] * stride;
            val inCoords = outCoords.clone();
            val deltaValue = delta[outCoords];
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winSize; rx++) {
                    val x = i - padding + rx;
                    if (x < 0 || x >= inShape[dims - 1]) continue;
                    val kcrx = new int[] { k, c, rx };
                    inCoords[dims - 2] = c;
                    inCoords[dims - 1] = x;
                    dX[inCoords] = dX[inCoords] + W[kcrx] * deltaValue;
                    dW[kcrx] = dW[kcrx] + input[inCoords] * deltaValue;
                }
            }
            db[k] = db[k] + delta[outCoords];
        });
        changes.incrementAndGet();
        return dX;
    }

    @Override
    public Conv1D update(@NonNull Optimizer optimizer, Regularizer regularizer) {
        var dW = this.dW / changes.get();
        var db = this.db / changes.get();
        if (Objects.nonNull(regularizer)) dW.copy(dW + regularizer.gradient(W));
        if (optimizer.paramCount() > 0) {
            val paramW = pW.computeIfAbsent(optimizer, //
                $ -> IntStream.range(0, optimizer.paramCount())//
                        .mapToObj(i -> NArray.like(W)).toArray(NArray[]::new)//
            );
            val paramB = pb.computeIfAbsent(optimizer, //
                $ -> IntStream.range(0, optimizer.paramCount())//
                        .mapToObj(i -> NArray.like(b)).toArray(NArray[]::new)//
            );
            W.copy(W + optimizer.update(dW, paramW));
            b.copy(b + optimizer.update(db, paramB));
        } else {
            W.copy(W + optimizer.update(dW, (NArray[]) null));
            b.copy(b + optimizer.update(db, (NArray[]) null));
        }
        clearGradients();
        return this;
    }

    public void clearGradients() {
        dW.fill(0.0);
        db.fill(0.0);
        changes.set(0);
    }

    @Override
    public int bytes() { return (W.size + b.size) * Double.BYTES; }

    @Override
    public DoubleBuffer parameterBuffer() {
        val buffer = ByteBuffer.allocateDirect((W.size + b.size) * Double.BYTES).asDoubleBuffer();
        return buffer.put(W.buffer()).put(b.buffer()).rewind();
    }

    @Override
    public Conv1D readParameter(DoubleBuffer buffer) {
        W.fill(buffer);
        b.fill(buffer);
        return this;
    }

    private int convolved(int inSize, int winSize, int stride, int padding) {
        int temp = inSize - winSize + padding * 2;
        if (temp % stride != 0) throw new IllegalArgumentException("Bad pooling size. input size = " + inSize
            + "window size= " + winSize + " stride = " + stride + " padding = " + padding);
        return temp / stride + 1;
    }

    @Override
    public String toString() {
        return "Conv layer ( filters = " + filterCount + ", channels " + channels + ", window = " + winSize
            + ", stride = " + stride + ", padding = " + padding + " )";
    }

}
