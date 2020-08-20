package hy.layer.feedforward;

import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class AvgPool implements FeedForwardLayer {
    int winWidth;
    int winHeight;
    int strideX;
    int strideY;
    int winVolume;

    @Builder
    public AvgPool(int winWidth, int winHeight, int strideX, int strideY) {
        this.winWidth = winWidth < 1 ? 2 : winWidth;
        this.winHeight = winHeight < 1 ? 2 : winHeight;
        this.strideX = strideX < 1 ? 1 : strideX;
        this.strideY = strideY < 1 ? 1 : strideY;
        this.winVolume = winWidth * winHeight;
    }

    public AvgPool(int winSize, int stride) { this(winSize, winSize, stride, stride); }

    public AvgPool(int winSize) { this(winSize, winSize, 1, 1); }

    @Override
    public NArray of(@NonNull NArray input, boolean isTraining) {
        val dims = input.dims;
        if (dims < 2) throw new IllegalArgumentException("Input dimension sshould be atleast 2. Got " + dims);
        val inShape = input.getShape();
        val outShape = inShape.clone();
        outShape[dims - 2] = pooled(inShape[dims - 2], winWidth, strideX);
        outShape[dims - 1] = pooled(inShape[dims - 1], winHeight, strideY);
        val out = new NArray(outShape).fill(outCoords -> {
            val i = outCoords[dims - 2] * strideX;
            val j = outCoords[dims - 1] * strideY;
            val inCoords = outCoords.clone();
            var sum = 0.0;
            for (var rx = 0; rx < winWidth; rx++) {
                inCoords[dims - 2] = i + rx;
                for (int ry = 0; ry < winHeight; ry++) {
                    inCoords[dims - 1] = j + ry;
                    sum += input[inCoords];
                }
            }
            return sum / winVolume;
        });
        return out;
    }

    @Override
    public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
        val inShape = input.getShape();
        val dims = inShape.length;
        val dX = NArray.like(input);
        delta.coordinates().forEach(outCoords -> {
            val i = outCoords[dims - 2] * strideX;
            val j = outCoords[dims - 1] * strideY;
            val inCoords = outCoords.clone();
            val deltaValue = delta[outCoords] / winVolume;
            for (var rx = 0; rx < winWidth; rx++) {
                inCoords[dims - 2] = i + rx;
                for (int ry = 0; ry < winHeight; ry++) {
                    inCoords[dims - 1] = j + ry;
                    dX[inCoords] = dX[inCoords] + deltaValue;
                }
            }
        });
        return dX;
    }

    private int pooled(int inSize, int winSize, int stride) {
        int temp = inSize - winSize;
        if (temp % stride != 0) throw new IllegalArgumentException(
                "Bad pooling size. input size =" + inSize + "window size= " + winSize + " & stride = " + stride);
        return temp / stride + 1;
    }

    @Override
    public String toString() {
        return "Avg Pooling (window = " + winWidth + " * " + winHeight + ", stride = [" + strideX + "," + strideY
            + "] )";
    }
}
