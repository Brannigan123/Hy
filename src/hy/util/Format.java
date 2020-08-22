package hy.util;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.NavigableMap;
import java.util.TreeMap;

import lombok.val;
import lombok.var;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Format {

    public String fmtDecimal(double value) { return toString(value); }

    public String fmtFraction(long value, long max) { return toString(value) + "/" + toString(max); }

    public String fmtFractionBar(int barLength, long value, long total) {
        return fmtFractionBar(barLength, (double) value / (double) total);
    }

    public String fmtFractionBar(int barLength, double value) {
        val bar = new char[barLength];
        val barIndex = (int) (barLength * value);

        Arrays.fill(bar, 0, barIndex, '\u2588');
        Arrays.fill(bar, barIndex, barLength, ' ');

        return String.copyValueOf(bar);
    }

    private NavigableMap<Double, String> magnitudeSuffixes = new TreeMap<>();

    static {
        magnitudeSuffixes[1e-24d] = "y"; // yocto
        magnitudeSuffixes[1e-21d] = "z"; // zepto
        magnitudeSuffixes[1e-18d] = "a"; // atto
        magnitudeSuffixes[1e-15d] = "f"; // femto
        magnitudeSuffixes[1e-12d] = "p"; // pico
        magnitudeSuffixes[1e-9d] = "n"; // nano
        magnitudeSuffixes[1e-6d] = "u"; // micro
        magnitudeSuffixes[1e-3d] = "m"; // milli
        magnitudeSuffixes[1e3d] = "k"; // kilo
        magnitudeSuffixes[1e6d] = "M"; // Mega
        magnitudeSuffixes[1e9d] = "G"; // Giga
        magnitudeSuffixes[1e12d] = "T"; // Tera
        magnitudeSuffixes[1e15d] = "P"; // Peta
        magnitudeSuffixes[1e18d] = "E"; // Exa
        magnitudeSuffixes[1e21d] = "Z"; // Zetta
        magnitudeSuffixes[1e24d] = "Y"; // Yotta
    }

    final DecimalFormat df2 = new DecimalFormat("#.##");
    final DecimalFormat df3 = new DecimalFormat("#.###");

    public String toString(double value) {
        if (value < 0d) return "-" + toString(-value);
        if (value == 0 || (value - 1e-4d > 1e-3d && value + 1e-4d < 1e3d)) return df3.format(value);

        val priorDivideBy = magnitudeSuffixes.floorKey(value);
        val priorRoundedScaled = Math.round(value * 1e2d / priorDivideBy) * 1e-2d;
        val rounded = priorRoundedScaled * priorDivideBy;

        val entry = magnitudeSuffixes.floorEntry(rounded);
        val divideBy = entry.getKey();
        val suffix = entry.getValue();

        val scaledRounded = rounded / divideBy;
        val scaledTruncated = (long) scaledRounded;
        val hasDecimal = (scaledRounded - scaledTruncated) >= 1e-2d;

        return hasDecimal ? df2.format(scaledRounded) + suffix : scaledTruncated + suffix;
    }

    public static void main(String[] args) {
        for (var i = 1d / 1_000_000_000_000d; i < 1_000_000_000_000d; i *= 10.5) {
            System.out.println(toString(i) + "\t" + i);
        }
    }
}
