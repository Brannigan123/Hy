package hy.layer;

import hy.util.NArray;

public interface Layer {
	public NArray of(NArray input, boolean isTraining);
	public NArray delta(NArray input, NArray output, NArray delta);
}
