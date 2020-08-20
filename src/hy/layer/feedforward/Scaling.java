package hy.layer.feedforward;

import hy.util.NArray;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;

@Value
@Builder
@AllArgsConstructor
public class Scaling implements FeedForwardLayer {

	double factor;

	@Override
	public NArray of(@NonNull NArray input, boolean isTraining) { return factor * input; }

	@Override
	public NArray delta(NArray input, NArray output, @NonNull NArray delta) { return factor * delta; }

	@Override
	public String toString() { return "Scaling (factor = " + factor + ")"; }

}
