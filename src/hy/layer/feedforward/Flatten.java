package hy.layer.feedforward;

import hy.util.NArray;
import lombok.NonNull;

public class Flatten implements FeedForwardLayer {

	@Override
	public NArray of(@NonNull NArray input, boolean isTraining) { return input.flattened(); }

	@Override
	public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
		return delta.reshaped(input.getShape());
	}

	@Override
	public String toString() { return "Flatten"; }
}
