package hy.layer.feedforward;

import static java.lang.System.arraycopy;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.WeakHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

import hy.optimizer.Optimizer;
import hy.regularizer.Regularizer;
import hy.util.NArray;
import lombok.AccessLevel;
import lombok.Builder;
import lombok.NonNull;
import lombok.val;
import lombok.var;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true, level = AccessLevel.PRIVATE)
public class FC implements ParamFeedForwardLayer {

    NArray                   W;
    NArray                   b;

    NArray                   dW;
    NArray                   db;

    Map<Optimizer, NArray[]> pW;
    Map<Optimizer, NArray[]> pb;

    AtomicLong               changes;

    @Builder
    public FC(int in, @NonNull int... out) {
        val dimW = new int[out.length + 1];
        dimW[0] = in;
        arraycopy(out, 0, dimW, 1, out.length);
        W = new NArray(dimW).randomize();
        b = new NArray(1).randomize();
        dW = NArray.like(W);
        db = NArray.like(b);
        pW = new WeakHashMap<>();
        pb = new WeakHashMap<>();
        changes = new AtomicLong();
    }

    @Override
    public NArray of(NArray input, boolean isTraining) { return input.dot(W) + b; }

    @Override
    public NArray delta(@NonNull NArray input, NArray output, @NonNull NArray delta) {
        val dX = NArray.like(input);
        val I = input.lastDim(), J = W.size / I, Z = input.size / I;
        for (var i = 0; i < I; i++) {
            for (var j = 0; j < J; j++) {
                val ij = new int[] { i, j };
                val W_ij = W[i * J + j];
                IntStream.range(0, Z)//
                        .forEach(z -> {
                            val in_zi = input[z * I + ij[0]];
                            val delta_zj = delta[z * J + ij[1]];
                            dX[z * I + ij[0]] = dX[z * I + ij[0]] + W_ij * delta_zj;
                            dW[ij[0] * J + ij[1]] = dW[ij[0] * J + ij[1]] + in_zi * delta_zj;
                            if (ij[0] == 0 && ij[1] == ij[0]) db[0] = db[0] + delta_zj;
                        });
            }
        }
        changes.incrementAndGet();
        return dX;
    }

    @Override
    public FC update(@NonNull Optimizer optimizer, Regularizer regularizer) {
        var dW = this.dW / changes.get();
        var db = this.db / changes.get();
        if (Objects.nonNull(regularizer)) dW.copy(dW + regularizer.gradient(W));
        if (optimizer.paramCount() > 0) {
            val paramW = pW.computeIfAbsent(optimizer, //
                $ -> IntStream.range(0, optimizer.paramCount())//
                        .mapToObj(i -> NArray.like(W)).toArray(NArray[]::new)//
            );
            val paramB = pb.computeIfAbsent(optimizer, //
                $ -> IntStream.range(0, optimizer.paramCount())//
                        .mapToObj(i -> NArray.like(b)).toArray(NArray[]::new)//
            );
            W.copy(W + optimizer.update(dW, paramW));
            b.copy(b + optimizer.update(db, paramB));
        } else {
            W.copy(W + optimizer.update(dW, (NArray[]) null));
            b.copy(b + optimizer.update(db, (NArray[]) null));
        }
        clearGradients();
        return this;
    }

    public void clearGradients() {
        dW.fill(0.0);
        db.fill(0.0);
        changes.set(0);
    }

    @Override
    public int bytes() { return (W.size + b.size) * Double.BYTES; }

    @Override
    public DoubleBuffer parameterBuffer() {
        val buffer = ByteBuffer.allocateDirect((W.size + b.size) * Double.BYTES).asDoubleBuffer();
        return buffer.put(W.buffer()).put(b.buffer()).rewind();
    }

    @Override
    public FC readParameter(DoubleBuffer buffer) {
        W.fill(buffer);
        b.fill(buffer);
        return this;
    }

    @Override
    public String toString() {
        val shape = W.getShape();
        val in = shape[0];
        val out = Arrays.copyOfRange(shape, 1, shape.length);
        return "FC layer ( " + in + "->" + Arrays.toString(out) + " )";
    }

}
