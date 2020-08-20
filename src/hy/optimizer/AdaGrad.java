package hy.optimizer;

import static hy.optimizer.GradOptimizer.Direction.DESCEND;

import hy.util.Epsilon;
import hy.util.NArray;
import lombok.AccessLevel;
import lombok.EqualsAndHashCode;
import lombok.NonNull;
import lombok.val;
import lombok.experimental.FieldDefaults;
import lombok.experimental.SuperBuilder;

@SuperBuilder
@EqualsAndHashCode(callSuper = false)
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class AdaGrad extends GradOptimizer {

	public AdaGrad(double learningRate, @NonNull Direction direction) { super(learningRate, direction); }

	public AdaGrad(double learningRate) { this(learningRate, Direction.DESCEND); }

	public AdaGrad(Direction direction) { super(direction); }

	@Override
	public int paramCount() { return 1; }

	@Override
	public void advance() {}

	@Override
	public NArray update(NArray grad, NArray... params) {
		val epsilon = Epsilon.get(), lr = direction == DESCEND ? -learningRate : learningRate;
		params[0].copy(params[0].bimap((v, g) -> v + g * g, grad));
		return params[0].bimap((v, g) -> lr * g / (Math.sqrt(v) + epsilon), grad);
	}

}
