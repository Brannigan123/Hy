package hy.util;

import lombok.experimental.UtilityClass;

@UtilityClass
public class Epsilon {

	private double value = 1e-8;// 1.11022302462515654042E-16;

	public double get() { return value; }

	public void set(double epsilon) {
		if (epsilon < 0) throw new IllegalArgumentException(
				"Epsilon can't be set a negative. Expected a magnitude value.");
		if (epsilon > 1) throw new IllegalArgumentException(
				"Epsilon should be a small value. Expected a fractional value");
		if (Double.isFinite(epsilon)) Epsilon.value = epsilon;
		else throw new IllegalArgumentException("Epsilon should be a finite value. Found " + epsilon);
	}
}
