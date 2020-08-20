package hy.layer.feedforward;

import static java.lang.System.arraycopy;

import java.util.Arrays;

import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class Reshape implements FeedForwardLayer {

	int   fromDim;
	int   outDims;
	int[] shape;

	@Builder
	public Reshape(int fromDim, @NonNull int... shape) {
		this.shape = shape.clone();
		this.fromDim = fromDim;
		this.outDims = fromDim + shape.length;
		if (shape.length == 0) throw new IllegalArgumentException("Shape must be provided.");
	}

	@Override
	public NArray of(@NonNull NArray input, boolean isTraining) {
		val origDims = input.dims;
		if (fromDim >= origDims)
			throw new IllegalArgumentException("Input is of a lower dimension than expected " + origDims);
		val outShape = new int[outDims];
		arraycopy(input.getShape(), 0, outShape, 0, fromDim);
		arraycopy(shape, 0, outShape, fromDim, shape.length);
		return input.reshaped(outShape);
	}

	@Override
	public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
		return delta.reshaped(input.getShape());
	}

	@Override
	public String toString() {
		return "Reshape (from dim " + fromDim + ", shape " + Arrays.toString(shape) + ")";
	}
}
