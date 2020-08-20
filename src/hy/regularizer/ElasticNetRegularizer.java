package hy.regularizer;

import hy.util.NArray;
import lombok.Builder;
import lombok.Builder.Default;
import lombok.experimental.FieldDefaults;

@Builder
@FieldDefaults(makeFinal = true)
public class ElasticNetRegularizer implements Regularizer {

	public static final ElasticNetRegularizer Default   = new ElasticNetRegularizer();

	@Default double                           L1_Lambda = 0.001;
	@Default double                           L2_Lambda = 0.001;

	public ElasticNetRegularizer(double l1_lambda, double l2_lambda) {
		this.L1_Lambda = l1_lambda;
		this.L2_Lambda = l2_lambda;
	}

	public ElasticNetRegularizer() {
		L1_Lambda = 0.001;
		L2_Lambda = 0.001;
	}

	public static ElasticNetRegularizer lambda(double l1_lambda, double l2_lambda) {
		return new ElasticNetRegularizer(l1_lambda, l2_lambda);
	}

	@Override
	public NArray gradient(NArray arr) { return arr.map(x -> L1_Lambda * Math.signum(x) + L2_Lambda * x); }

}
