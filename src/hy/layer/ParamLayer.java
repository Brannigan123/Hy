package hy.layer;

import java.nio.DoubleBuffer;

import hy.optimizer.Optimizer;
import hy.regularizer.Regularizer;
import lombok.val;

public interface ParamLayer extends Layer {
    public void clearGradients();
    public ParamLayer update(Optimizer optimizer, Regularizer regularizer);
    public int bytes();
    public DoubleBuffer parameterBuffer();
    public ParamLayer readParameter(DoubleBuffer buffer);

    public default ParamStatistics paramStatistics() {
        val stats = new ParamStatistics();
        val buf = parameterBuffer();
        while (buf.hasRemaining()) stats.accept(buf.get());
        return stats;
    }
}
