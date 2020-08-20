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
import io.vavr.API;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import lombok.var;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class TConv implements ParamFeedForwardLayer {

    int                      filterCount;
    int                      channels;
    int                      winWidth;
    int                      winHeight;
    int                      strideX;
    int                      strideY;
    int                      trimX;
    int                      trimY;

    NArray                   W;
    NArray                   b;

    NArray                   dW;
    NArray                   db;

    Map<Optimizer, NArray[]> pW;
    Map<Optimizer, NArray[]> pb;

    AtomicLong               changes;

    @Builder
    public TConv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY, int trimX,
        int trimY) {
        this.filterCount = filterCount < 1 ? 1 : filterCount;
        this.channels = channels < 1 ? 1 : channels;
        this.winWidth = winWidth < 1 ? 1 : winWidth;
        this.winHeight = winHeight < 1 ? 1 : winHeight;
        this.strideX = strideX < 1 ? 1 : strideX;
        this.strideY = strideY < 1 ? 1 : strideY;
        this.trimX = trimX < 0 ? 0 : trimX;
        this.trimY = trimY < 0 ? 0 : trimY;
        W = new NArray(filterCount, channels, winWidth, winHeight).randomize();
        b = new NArray(filterCount).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
    }

    public TConv(int filterCount, int channels, int winSize, int stride, int trim) {
        this(filterCount, channels, winSize, winSize, stride, stride, trim, trim);
    }

    public TConv(int filterCount, int channels, int winSize, int trim) {
        this(filterCount, channels, winSize, winSize, 1, 1, trim, trim);
    }

    public TConv(int filterCount, int channels, int winWidth, int winHeight, int strideX, int strideY,
        @NonNull Padding trim) {
        this.filterCount = filterCount < 1 ? 1 : filterCount;
        this.channels = channels < 1 ? 1 : channels;
        this.winWidth = winWidth < 1 ? 1 : winWidth;
        this.winHeight = winHeight < 1 ? 1 : winHeight;
        this.strideX = strideX < 1 ? 1 : strideX;
        this.strideY = strideY < 1 ? 1 : strideY;
        W = new NArray(filterCount, channels, winWidth, winHeight).randomize();
        b = new NArray(filterCount).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
        switch (trim) {
        case SAME:
            if ((winWidth - 1) % 2 != 0)
                throw new IllegalArgumentException("Bad convolution parameters. Filter width should be odd.");
            if ((winHeight - 1) % 2 != 0)
                throw new IllegalArgumentException("Bad convolution parameters. Filter height should be odd.");
            this.trimX = (winWidth - 1) / 2;
            this.trimY = (winHeight - 1) / 2;
            break;
        case VALID:
        default:
            this.trimX = 0;
            this.trimY = 0;
        }
    }

    public TConv(int filterCount, int channels, int winSize, int stride, Padding trim) {
        this(filterCount, channels, winSize, winSize, stride, stride, trim);
    }

    public TConv(int filterCount, int channels, int winSize, Padding trim) {
        this(filterCount, channels, winSize, winSize, 1, 1, trim);
    }

    public TConv(int filterCount, int channels, int winSize) {
        this(filterCount, channels, winSize, winSize, 1, 1, 0, 0);
    }

    @Override
    public NArray of(@NonNull NArray input, boolean isTraining) {
        val dims = input.dims;
        if (dims < 3) throw new IllegalArgumentException("Input dimensions should be atleast 3. Got " + dims);
        val inShape = input.getShape();
        val outShape = inShape.clone();
        outShape[dims - 3] = filterCount;
        outShape[dims - 2] = transConvolved(inShape[dims - 2], winWidth, strideX, trimX);
        outShape[dims - 1] = transConvolved(inShape[dims - 1], winHeight, strideY, trimY);
        val out = new NArray(outShape).fill(outCoords -> {
            val k = outCoords[dims - 3];
            val i = outCoords[dims - 2] / strideX;
            val j = outCoords[dims - 1] / strideY;
            val inCoords = outCoords.clone();
            var sum = 0.0;
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winWidth; rx++) {
                    val x = i + trimX - rx;
                    if (x < 0 || x >= inShape[dims - 2]) continue;
                    for (int ry = 0; ry < winHeight; ry++) {
                        val y = j + trimY - ry;
                        if (y < 0 || y >= inShape[dims - 1]) continue;
                        val kcrxry = new int[] { k, c, rx, ry };
                        inCoords[dims - 3] = c;
                        inCoords[dims - 2] = x;
                        inCoords[dims - 1] = y;
                        sum = sum + input[inCoords] * W[kcrxry];
                    }
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
            val k = outCoords[dims - 3];
            val i = outCoords[dims - 2] / strideX;
            val j = outCoords[dims - 1] / strideY;
            val inCoords = outCoords.clone();
            val deltaValue = delta[outCoords];
            for (var c = 0; c < channels; c++) {
                for (var rx = 0; rx < winWidth; rx++) {
                    val x = i + trimX - rx;
                    if (x < 0 || x >= inShape[dims - 2]) continue;
                    for (int ry = 0; ry < winHeight; ry++) {
                        val y = j + trimY - ry;
                        if (y < 0 || y >= inShape[dims - 1]) continue;
                        val kcrxry = new int[] { k, c, rx, ry };
                        inCoords[dims - 3] = c;
                        inCoords[dims - 2] = x;
                        inCoords[dims - 1] = y;
                        dX[inCoords] = dX[inCoords] + W[kcrxry] * deltaValue;
                        dW[kcrxry] = dW[kcrxry] + input[inCoords] * deltaValue;
                    }
                }
            }
            db[k] = db[k] + delta[outCoords];
        });
        changes.incrementAndGet();
        return dX;
    }

    @Override
    public TConv update(@NonNull Optimizer optimizer, Regularizer regularizer) {
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
    public TConv readParameter(DoubleBuffer buffer) {
        W.fill(buffer);
        b.fill(buffer);
        return this;
    }

    private static int transConvolved(int inSize, int winSize, int stride, int trim) {
        return (inSize - 1) * stride + winSize - trim * 2;
    }

    @Override
    public String toString() {
        return "TConv layer ( filters = " + filterCount + ", channels " + channels + ", window = [" + winWidth + ","
            + winHeight + "], strides = [" + strideX + "," + strideY + "]" + ", trim = [" + trimX + "," + trimY + "] )";
    }

    public static void main(String[] args) { API.println(transConvolved(253, 4, 1, 0)); }
}
