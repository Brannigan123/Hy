package hy.layer.feedforward;

import hy.util.NArray;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;

@Value
public class Dropout implements FeedForwardLayer {

	double rate;
	double scale;

	@Builder
	public Dropout(double rate) {
		this.rate = rate;
		this.scale = 1.0 / (1.0 - rate);
	}

	@Override
	public NArray of(@NonNull NArray input, boolean isTraining) {
		if (isTraining) return input.map(x -> Math.random() < rate ? 0.0 : scale * x);
		return input;
	}

	@Override
	public NArray delta(@NonNull NArray input, @NonNull NArray output, @NonNull NArray delta) {
		return NArray.like(input)
				.fill(coords -> output[coords] != 0.0 || input[coords] == 0.0 ? scale * delta[coords] : 0.0);
	}

	@Override
	public String toString() { return "Dropout (rate=" + rate + ")"; }
}
