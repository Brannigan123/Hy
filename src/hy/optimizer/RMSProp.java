package hy.optimizer;

import static hy.optimizer.GradOptimizer.Direction.DESCEND;

import hy.util.Epsilon;
import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder.Default;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import lombok.val;
import lombok.experimental.FieldDefaults;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class RMSProp extends GradOptimizer {

	@Getter @Default double mu = 0.9;

	public RMSProp(double learningRate, @NonNull Direction direction, double mu) {
		super(learningRate, direction);
		this.mu = mu;
	}

	public RMSProp(double learningRate, @NonNull Direction direction) {
		super(learningRate, direction);
		this.mu = 0.9;
	}

	public RMSProp(double learningRate, double mu) {
		super(learningRate);
		this.mu = mu;
	}

	public RMSProp(double learningRate) {
		super(learningRate);
		this.mu = 0.9;
	}

	public RMSProp(@NonNull Direction direction) {
		super(direction);
		this.mu = 0.9;
	}

	@Override
	public int paramCount() { return 1; }

	@Override
	public void advance() {}

	@Override
	public NArray update(NArray grad, NArray... params) {
		val epsilon = Epsilon.get(), lr = direction == DESCEND ? -learningRate : learningRate;
		params[0].copy(params[0].bimap((v, g) -> v * mu + g * g * (1.0 - mu), grad));
		return params[0].bimap((v, g) -> lr * g / (Math.sqrt(v) + epsilon), grad);
	}

}
