package hy.optimizer;

import lombok.AccessLevel;
import lombok.Builder.Default;
import lombok.Data;
import lombok.NonNull;
import lombok.experimental.Accessors;
import lombok.experimental.FieldDefaults;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@Accessors(chain = true)
@FieldDefaults(level = AccessLevel.PROTECTED)
public abstract class GradOptimizer implements Optimizer {

	@Default double             learningRate = 1e-3;
	@Default @NonNull Direction direction    = Direction.DESCEND;

	public GradOptimizer(double learningRate, @NonNull Direction direction) {
		this.learningRate = checkedLearningRate(learningRate);
		this.direction = direction;
	}

	public GradOptimizer(double learningRate) {
		this.learningRate = checkedLearningRate(learningRate);
		this.direction = Direction.DESCEND;
	}

	public GradOptimizer(@NonNull Direction direction) {
		this.learningRate = 1e-3;
		this.direction = direction;
	}

	public GradOptimizer setLearningRate(double learningRate) {
		this.learningRate = checkedLearningRate(learningRate);
		return this;
	}

	protected double checkedLearningRate(double learningRate) {
		if (learningRate < 0) throw new IllegalArgumentException("Negative learning rate supplied.");
		return learningRate;
	}

	public static enum Direction {
		ASCEND, DESCEND
	}
}
