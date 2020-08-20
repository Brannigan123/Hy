package hy.regularizer;

import hy.util.NArray;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.experimental.FieldDefaults;

@Builder
@FieldDefaults(makeFinal = true)
public class L1Regularizer implements Regularizer {

	public static final L1Regularizer Default = new L1Regularizer();

	@Default double                   lambda  = 0.01;

	public L1Regularizer(double lambda) { this.lambda = lambda; }

	public L1Regularizer() { lambda = 0.01; }

	public static L1Regularizer lambda(double lambda) { return new L1Regularizer(lambda); }

	@Override
	public NArray gradient(NArray arr) { return arr.map(x -> lambda * Math.signum(x)); }

}
