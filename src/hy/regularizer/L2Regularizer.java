package hy.regularizer;

import hy.util.NArray;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.experimental.FieldDefaults;

@Builder
@FieldDefaults(makeFinal = true)
public class L2Regularizer implements Regularizer {

	public static final L2Regularizer Default = new L2Regularizer();

	@Default double                   lambda  = 0.01;

	public L2Regularizer(double lambda) { this.lambda = lambda; }

	public L2Regularizer() { lambda = 0.01; }

	public static L2Regularizer lambda(double lambda) { return new L2Regularizer(lambda); }

	@Override
	public NArray gradient(NArray arr) { return lambda * arr; }

}
