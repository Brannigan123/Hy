package hy.optimizer;

import static hy.optimizer.GradOptimizer.Direction.DESCEND;

import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder.Default;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.NonNull;
import lombok.experimental.FieldDefaults;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class NAG extends GradOptimizer {

	@Getter @Default double mu = 0.9;

	public NAG(double learningRate, @NonNull Direction direction, double mu) {
		super(learningRate, direction);
		this.mu = mu;
	}

	public NAG(double learningRate, @NonNull Direction direction) {
		super(learningRate, direction);
		this.mu = 0.9;
	}

	public NAG(double learningRate, double mu) {
		super(learningRate);
		this.mu = mu;
	}

	public NAG(double learningRate) {
		super(learningRate);
		this.mu = 0.9;
	}

	public NAG(@NonNull Direction direction) {
		super(direction);
		this.mu = 0.9;
	}

	@Override
	public int paramCount() { return 1; }

	@Override
	public void advance() {}

	@Override
	public NArray update(NArray grad, NArray... params) {
		params[0].copy(params[0].bimap((v, g) -> v * mu - g * learningRate, grad));
		return direction == DESCEND ? params[0] : -params[0];
	}

}
