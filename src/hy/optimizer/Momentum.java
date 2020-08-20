package hy.optimizer;

import static hy.optimizer.GradOptimizer.Direction.DESCEND;

import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder.Default;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.val;
import lombok.experimental.FieldDefaults;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class Momentum extends GradOptimizer {

	@Getter @Default double mu = 0.9;

	public Momentum(double learningRate, Direction direction, double mu) {
		super(learningRate, direction);
		this.mu = mu;
	}

	public Momentum(double learningRate, Direction direction) {
		super(learningRate, direction);
		this.mu = 0.9;
	}

	public Momentum(double learningRate, double mu) {
		super(learningRate);
		this.mu = mu;
	}

	public Momentum(double learningRate) {
		super(learningRate);
		this.mu = 0.9;
	}

	public Momentum(Direction direction) {
		super(direction);
		this.mu = 0.9;
	}

	@Override
	public int paramCount() { return 1; }

	@Override
	public void advance() {}

	@Override
	public NArray update(NArray grad, NArray... params) {
		val prevv = params[0];
		params[0].copy(params[0].bimap((v, g) -> v * mu - g * learningRate, grad));
		val delta = prevv.bimap((pv, v) -> pv * mu - v * (1.0 + mu), params[0]);
		return direction == DESCEND ? -delta : delta;
	}

}
