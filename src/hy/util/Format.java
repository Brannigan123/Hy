package hy.util;

import java.util.Arrays;

import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Format {

    public String fmtDecimal(double value) { return String.format("%20s", String.valueOf(value)); }

    public String fmtFraction(long value, long total) { return fmtWhole(value, total) + "/" + total; }

    public String fmtWhole(long value, long max) {
        val maxEpochStr = String.valueOf(max);
        val epochStr = String.format("%" + (maxEpochStr.length()) + "d", value);
        return epochStr;
    }

    public String fmtFractionBar(int barLength, long value, long total) {
        val bar = new char[barLength];
        val barIndex = (int) (barLength * value / total);
        Arrays.fill(bar, 0, barIndex, '\u2588');
        Arrays.fill(bar, barIndex, barLength, ' ');
        return String.copyValueOf(bar);
    }

    public String fmtFractionBar(int barLength, double value) {
        val bar = new char[barLength];
        val barIndex = (int) (barLength * value);
        Arrays.fill(bar, 0, barIndex, '\u2588');
        Arrays.fill(bar, barIndex, barLength, ' ');
        return String.copyValueOf(bar);
    }
    
}
