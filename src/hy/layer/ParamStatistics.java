package hy.layer;

import java.util.function.DoubleConsumer;

import hy.util.Epsilon;
import lombok.AccessLevel;
import lombok.experimental.FieldDefaults;

@FieldDefaults(level = AccessLevel.PRIVATE)
public class ParamStatistics implements DoubleConsumer {
    long   count;
    long   zerosCount;
    double sum;
    double sumCompensation;
    double simpleSum;
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;

    public ParamStatistics() {}

    @Override
    public void accept(double value) {
        ++count;
        if (Math.abs(value) <= Epsilon.get()) ++zerosCount;
        simpleSum += value;
        sumWithCompensation(value);
        min = Math.min(min, value);
        max = Math.max(max, value);
    }

    public ParamStatistics accept(ParamStatistics other) {
        count += other.count;
        zerosCount += other.zerosCount;
        simpleSum += other.simpleSum;
        sumWithCompensation(other.sum);
        sumWithCompensation(other.sumCompensation);
        min = Math.min(min, other.min);
        max = Math.max(max, other.max);
        return this;
    }

    private void sumWithCompensation(double value) {
        double tmp = value - sumCompensation;
        double velvel = sum + tmp;
        sumCompensation = (velvel - sum) - tmp;
        sum = velvel;
    }

    public final long getCount() { return count; }

    public final long getZerosCount() { return zerosCount; }

    public final double getSum() {
        double tmp = sum + sumCompensation;
        if (Double.isNaN(tmp) && Double.isInfinite(simpleSum)) return simpleSum;
        else return tmp;
    }

    public final double getMin() { return min; }

    public final double getMax() { return max; }

    public final double getAverage() { return getCount() > 0 ? getSum() / getCount() : 0.0; }

    @Override
    public String toString() {
        return String.format("%s{count=%d, zeros=%d, sum=%f, min=%f, average=%f, max=%f}",
            this.getClass().getSimpleName(), getCount(), getZerosCount(), getSum(), getMin(), getAverage(),
            getMax());
    }

}
