package hy.regularizer;

import hy.util.NArray;

public interface Regularizer {
	public NArray gradient(NArray arr);
}
