package hy.optimizer;

import static hy.optimizer.GradOptimizer.Direction.DESCEND;

import hy.util.NArray;
import lombok.val;
import lombok.experimental.SuperBuilder;

@SuperBuilder
public class SGD extends GradOptimizer {

	public SGD(double learningRate, Direction direction) { super(learningRate, direction); }

	public SGD(double learningRate) { super(learningRate); }

	public SGD(Direction direction) { super(direction); }

	@Override
	public int paramCount() { return 0; }

	@Override
	public void advance() {}

	@Override
	public NArray update(NArray grad, NArray... params) {
		val lr = direction == DESCEND ? -learningRate : learningRate;
		return lr * grad;
	}

}
