package hy.optimizer;

import hy.util.NArray;

public interface Optimizer {
	public int paramCount();
	public void advance();
	public NArray update(NArray grad, NArray... params);
}
