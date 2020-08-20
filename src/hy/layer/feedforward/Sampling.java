package hy.layer.feedforward;

import java.util.Arrays;

import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.FieldDefaults;

@Builder
@RequiredArgsConstructor(access = AccessLevel.PUBLIC)
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class Sampling implements FeedForwardLayer {

    @NonNull Distribution distribution;

    @Override
    public NArray of(@NonNull NArray input, boolean isTraining) {
        val inDims = input.dims, outDims = inDims - 1;
        val inShape = input.getShape();
        val outShape = Arrays.copyOf(inShape, outDims);
        if (inShape[outDims] != 2)
            throw new IllegalArgumentException("Sampling layer expects last dimension to be of cardinality 2.");
        val out = new NArray(outShape).fill(outCoords -> {
            val meanCoords = Arrays.copyOf(outCoords, outDims);
            val stdCoords = Arrays.copyOf(outCoords, outDims);
            stdCoords[outDims] = 1;
            return input[meanCoords] + Math.exp(input[stdCoords]) * distribution.poll();
        });
        return out;
    }

    // Regularization term : -0.5*mean(1+mu-sqr(sigma)0exp(sigma))
    @Override
    public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
        val inDims = input.dims, outDims = inDims - 1;
        val dX = NArray.like(input).fill(inCoords -> {
            val outCoords = Arrays.copyOf(inCoords, outDims);
            if (inCoords[outDims] == 1) {
                val meanCoords = inCoords.clone();
                meanCoords[outDims] = 0;
                val mean = input[meanCoords];
                val std = input[inCoords];
                val expStd = Math.exp(std);
                val epsilonExpStd = (output[outCoords] - mean);
                return epsilonExpStd * delta[outCoords]/* regularization */ - 0.5 + 0.5 * expStd;
            }
            return delta[outCoords]/* regularization */ + input[inCoords];
        });
        return dX;
    }

    public static interface Distribution {
        public double poll();
    }

}
