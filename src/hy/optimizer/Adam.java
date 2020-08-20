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
@EqualsAndHashCode(callSuper = false, exclude = { "betaT" })
@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class Adam extends GradOptimizer {
	@Getter @Default double beta1 = 0.9;
	@Getter @Default double beta2 = 0.999;

	final double[]          betaT = new double[2];

	public Adam(double learningRate, @NonNull Direction direction, double beta1, double beta2) {
		super(learningRate, direction);
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.betaT[0] = beta1;
		this.betaT[1] = beta2;
	}

	public Adam(double learningRate, @NonNull Direction direction) {
		super(learningRate, direction);
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		this.betaT[0] = beta1;
		this.betaT[1] = beta2;
	}

	public Adam(double learningRate, double beta1, double beta2) {
		super(learningRate);
		this.beta1 = beta1;
		this.beta2 = beta2;
		this.betaT[0] = beta1;
		this.betaT[1] = beta2;
	}

	public Adam(double learningRate) {
		super(learningRate);
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		this.betaT[0] = beta1;
		this.betaT[1] = beta2;
	}

	public Adam(@NonNull Direction direction) {
		super(direction);
		this.beta1 = 0.9;
		this.beta2 = 0.999;
		this.betaT[0] = beta1;
		this.betaT[1] = beta2;
	}

	@Override
	public int paramCount() { return 2; }

	@Override
	public void advance() {
		betaT[0] *= beta1;
		betaT[1] *= beta2;
	}

	@Override
	public NArray update(NArray grad, NArray... params) {
		val epsilon = Epsilon.get(), lr = direction == DESCEND ? -learningRate : learningRate;
		params[0].copy(params[0].bimap((m, g) -> m * beta1 + g * (1.0 - beta1), grad));
		params[1].copy(params[1].bimap((v, g) -> v * beta2 + g * g * (1.0 - beta2), grad));
		return params[0].bimap( (m, v) -> lr * (m / (1 - betaT[0])) / (Math.sqrt(v / (1.0 - betaT[1])) + epsilon),
								params[1]);
	}

}
