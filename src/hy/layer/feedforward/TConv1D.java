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
public class TConv1D implements ParamFeedForwardLayer {

    int                      filterCount;
    int                      channels;
    int                      winSize;
    int                      stride;
    int                      trim;

    NArray                   W;
    NArray                   b;

    NArray                   dW;
    NArray                   db;

    Map<Optimizer, NArray[]> pW;
    Map<Optimizer, NArray[]> pb;

    AtomicLong               changes;

    @Builder
    public TConv1D(int filterCount, int channels, int winSize, int stride, int trim) {
        this.filterCount = filterCount < 1 ? 1 : filterCount;
        this.channels = channels < 1 ? 1 : channels;
        this.winSize = winSize < 1 ? 1 : winSize;
        this.stride = stride < 1 ? 1 : stride;
        this.trim = trim < 0 ? 0 : trim;
        W = new NArray(filterCount, channels, winSize).randomize();
        b = new NArray(filterCount).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
    }

    public TConv1D(int filterCount, int channels, int winSize, int trim) {
        this(filterCount, channels, winSize, 1, trim);
    }

    public TConv1D(int filterCount, int channels, int winSize, int stride, @NonNull Padding trim) {
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
        switch (trim) {
        case SAME:
            if ((winSize - 1) % 2 != 0)
                throw new IllegalArgumentException("Bad convolution parameters. Filter width should be odd.");
            this.trim = (winSize - 1) / 2;
            break;
        case VALID:
        default:
            this.trim = 0;
        }
    }

    public TConv1D(int filterCount, int channels, int winSize, Padding trim) {
        this(filterCount, channels, winSize, 1, trim);
    }

    public TConv1D(int filterCount, int channels, int winSize) { this(filterCount, channels, winSize, 1, 0); }

    @Override
    public NArray of(@NonNull NArray input, boolean isTraining) {
        val dims = input.dims;
        if (dims < 3) throw new IllegalArgumentException("Input dimensions should be atleast 3. Got " + dims);
        val inShape = input.getShape();
        val outShape = inShape.clone();
        outShape[dims - 2] = filterCount;
        outShape[dims - 1] = transConvolved(inShape[dims - 1], winSize, stride, trim);
        val out = new NArray(outShape).fill(outCoords -> {
            val k = outCoords[dims - 2];
            val i = outCoords[dims - 1] / stride;
            val inCoords = outCoords.clone();
            var sum = 0.0;
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winSize; rx++) {
                    val x = i + trim - rx;
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
            val i = outCoords[dims - 1] / stride;
            val inCoords = outCoords.clone();
            val deltaValue = delta[outCoords];
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winSize; rx++) {
                    val x = i + trim - rx;
                    if (x < 0 || x >= inShape[dims - 2]) continue;
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
    public TConv1D update(@NonNull Optimizer optimizer, Regularizer regularizer) {
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
    public TConv1D readParameter(DoubleBuffer buffer) {
        W.fill(buffer);
        b.fill(buffer);
        return this;
    }

    private int transConvolved(int inSize, int winSize, int stride, int trim) {
        return (inSize - 1) * stride + winSize - trim * 2;
    }

    @Override
    public String toString() {
        return "TConv layer ( filters = " + filterCount + ", channels " + channels + ", window = " + winSize + "*"
            + ", strides = " + stride + ", trim = " + trim + " )";
    }

}
