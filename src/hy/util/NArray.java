package hy.util;

import static java.lang.System.*;
import static java.lang.Math.*;
import static io.vavr.API.*;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Arrays;
import java.util.Iterator;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import io.vavr.Tuple2;
import io.vavr.collection.Array;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.NonNull;
import lombok.Value;
import lombok.val;
import lombok.var;
import lombok.experimental.FieldDefaults;

@FieldDefaults(makeFinal = true)
public class NArray implements Cloneable, Iterable<Double> {

    static Random rand = new Random();

    public int    size;
    public int    dims;
    public int    shapeSum;

    DoubleBuffer  buffer;
    int[]         shape;
    int[]         blockSize;

    @Builder
    public NArray(@NonNull int[] shape, @NonNull double... data) {
        this(shape);
        fill(data);
    }

    public NArray(int... shape) {
        this.shape = shape.clone();
        this.dims = shape.length;
        this.shapeSum = IntStream.of(shape).sum();
        this.blockSize = new int[dims];
        this.size = initSizes();
        this.buffer = ByteBuffer//
                .allocateDirect(Math.multiplyExact(size, Double.BYTES))//
                .asDoubleBuffer();
    }

    private int initSizes() {
        blockSize[dims - 1] = 1;
        var size = shape[dims - 1];
        for (int i = dims - 2; i >= 0; i--) {
            blockSize[i] = blockSize[i + 1] * shape[i + 1];
            size *= shape[i];
        }
        return size;
    }

    public static NArray of(@NonNull NArray... arrays) {
        val iShape = broadcast(Stream.of(arrays).map(NArray::getShape).toArray(int[][]::new));
        val shape = new int[iShape.length + 1];
        arraycopy(iShape, 0, shape, 1, iShape.length);
        shape[0] = arrays.length;
        val stack = new NArray(shape).fill(coords -> {
            val array = arrays[coords[0]];
            return array[array.clipCoords(coords)];
        });
        return stack;
    }

    public static NArray of(@NonNull double... data) { return new NArray(new int[] { data.length }, data); }

    public static NArray of(@NonNull double[][] data) {
        val d1 = data.length;
        val d2 = Stream.of(data).mapToInt(arr -> arr.length).max().orElse(0);
        val shape = new int[] { d1, d2 };
        val arr = new NArray(shape);
        int idx = 0;
        for (var i = 0; i < d1; i++) for (var j = 0; j < d2; j++) arr[idx++] = data[i][j];
        return arr;
    }

    public static NArray of(@NonNull double[][]... data) {
        val d1 = data.length;
        val d2 = Stream.of(data).mapToInt(arr -> arr.length).max().orElse(0);
        val d3 = Stream.of(data).mapToInt(arr2d -> Stream.of(arr2d).mapToInt(arr -> arr.length).max().orElse(0)).max()
                .orElse(0);
        val shape = new int[] { d1, d2, d3 };
        val arr = new NArray(shape);
        int idx = 0;
        for (int i = 0; i < d1; i++)
            for (int j = 0; j < d2; j++) for (int k = 0; k < d3; k++) arr[idx++] = data[i][j][k];
        return arr;
    }

    public static NArray of(@NonNull double[][][]... data) {
        val d1 = data.length;
        val d2 = Stream.of(data).mapToInt(arr -> arr.length).max().orElse(0);
        val d3 = Stream.of(data).mapToInt(arr2d -> Stream.of(arr2d).mapToInt(arr -> arr.length).max().orElse(0)).max()
                .orElse(0);
        val d4 = Stream.of(data)
                .mapToInt(arr3d -> Stream.of(arr3d)
                        .mapToInt(arr2d -> Stream.of(arr2d).mapToInt(arr -> arr.length).max().orElse(0)).max()
                        .orElse(0))
                .max().orElse(0);
        val shape = new int[] { d1, d2, d3, d4 };
        val arr = new NArray(shape);
        int idx = 0;
        for (int i = 0; i < d1; i++) for (int j = 0; j < d2; j++)
            for (int k = 0; k < d3; k++) for (int l = 0; l < d4; l++) arr[idx++] = data[i][j][k][l];
        return arr;
    }

    public static NArray like(NArray other) { return new NArray(other.shape); }

    public int indexOf(@NonNull int... coords) {
        val n = coords.length;
        if (n == 0) throw new IllegalArgumentException();
        var index = 0;
        for (int i = 0; i < Math.min(dims, n); i++) index += blockSize[i] * normCoord(coords[i], shape[i]);
        return index;
    }

    private int normCoord(int coord, int card) {
        if (coord > -card && coord < card) return coord < 0 ? coord + card : coord;
        if (card == 1) return 0;
        print(Arrays.toString(shape) + " " + coord);
        throw new IllegalArgumentException();

    }

    public int[] coordinatesOf(int index) {
        if (index < 0 || index >= size)
            throw new IllegalArgumentException("Index out of range: " + index + " for shape " + Arrays.toString(shape));
        val coordinates = new int[dims];
        for (int i = 0; i < dims; i++) {
            coordinates[i] = index / blockSize[i];
            index -= coordinates[i] * blockSize[i];
        }
        return coordinates;
    }

    public double get(int index) { return buffer[index]; }

    public double get(int... coords) { return buffer[indexOf(coords)]; }

    public NArray set(int[] coords, double value) {
        buffer[indexOf(coords)] = value;
        return this;
    }

    public NArray set(int index, double value) {
        buffer[index] = value;
        return this;
    }

    public NArray fill(double value) {
        for (var i = 0; i < size; i++) buffer[i] = value;
        return this;
    }

    public NArray fill(@NonNull double... data) {
        val n = Math.min(size, data.length);
        buffer.put(data, 0, n);
        return this;
    }

    public NArray fill(@NonNull DoubleBuffer data) {
        int p = data.position();
        int n = Math.min(size, data.limit() - p);
        for (int i = 0; i < n; i++) buffer[i] = data[p + i];
        data.position(p + n);
        return this;
    }

    public NArray fill(Function<int[], Double> src) {
        coordinates().parallel().forEach(coords -> this[coords] = src.apply(coords));
        return this;
    }

    public NArray copy(@NonNull NArray other) {
        fill(other.buffer());
        return this;
    }

    public NArray update(@NonNull DoubleUnaryOperator op) {
        IntStream.range(0, size).parallel().forEach(i -> buffer[i] = op.applyAsDouble(buffer[i]));
        return this;
    }

    public NArray randomize() {
        IntStream.range(0, size).parallel().forEach(i -> buffer[i] = rand.nextGaussian() * sqrt(2.0 / shapeSum));
        return this;
    }

    public double sum() { return reduce(0, Double::sum); }

    public double average() { return reduce(0, Double::sum) / size; }

    public double max() { return reduce(Double.MIN_VALUE, Math::max); }

    public double min() { return reduce(Double.MAX_VALUE, Math::min); }

    public int[] argmax() {
        return coordinatesOf(Array.ofAll(this).zipWithIndex()//
                .<Double>maxBy(Tuple2::_1)//
                .get()._2);
    }

    public int[] argmin() {
        return coordinatesOf(Array.ofAll(this).zipWithIndex()//
                .<Double>minBy(Tuple2::_1)//
                .get()._2);
    }

    public NArray negate() { return this.map(x -> -x); }

    public NArray add(NArray arg2) { return this.bimap(Double::sum, arg2); }

    public NArray add(double arg2) { return this.bimap(Double::sum, arg2); }

    public NArray addRev(double arg2) { return this.bimapRev(Double::sum, arg2); }

    public NArray subtract(NArray arg2) { return this.bimap((a, b) -> a - b, arg2); }

    public NArray subtract(double arg2) { return this.bimap((a, b) -> a - b, arg2); }

    public NArray subtractRev(double arg2) { return this.bimapRev((a, b) -> a - b, arg2); }

    public NArray multiply(NArray arg2) { return this.bimap((a, b) -> a * b, arg2); }

    public NArray multiply(double arg2) { return this.bimap((a, b) -> a * b, arg2); }

    public NArray multiplyRev(double arg2) { return this.bimapRev((a, b) -> a * b, arg2); }

    public NArray divide(NArray arg2) { return this.bimap((a, b) -> a / b, arg2); }

    public NArray divide(double arg2) { return this.bimap((a, b) -> a / b, arg2); }

    public NArray divideRev(double arg2) { return this.bimapRev((a, b) -> a / b, arg2); }

    public NArray dot(@NonNull NArray other) {
        val A = this.shape, B = other.shape;
        val M = A.length, N = B.length;
        val resultDims = M + N - 2;
        if (resultDims == 0) {
            val prod = this * other;
            return NArray.of(prod.values().sum());
        }
        val resultCards = new int[resultDims];
        arraycopy(A, 0, resultCards, 0, M - 1);
        arraycopy(B, 1, resultCards, M - 1, N - 1);
        val result = new NArray(resultCards);
        val C = dotCommon(A[M - 1], B[0]);
        result.coordinates().parallel().forEach(resultCoords -> {
            val SCoords = new int[M];
            val TCoords = new int[N];
            arraycopy(resultCoords, 0, SCoords, 0, M - 1);
            arraycopy(resultCoords, M - 1, TCoords, 1, N - 1);
            SCoords[M - 1] = TCoords[0] = 0;
            var sum = this[SCoords] * other[TCoords];
            for (var c = 1; c < C; c++) {
                SCoords[M - 1] = TCoords[0] = c;
                sum = sum + this[SCoords] * other[TCoords];
            }
            result[resultCoords] = sum;
        });
        return result;
    }

    private int dotCommon(int a, int b) {
        if (a == b) return a;
        throw new IllegalArgumentException(
                "Dimensions are incompatible. Last dimension of 1st array has to match first dimension of 2nd array");
    }

    public NArray T() {
        val arr = new NArray(reverse(shape));
        arr.coordinates().forEach(coords -> arr[coords] = this[reverse(coords)]);
        return arr;
    }

    private int[] reverse(int... original) {
        val n = original.length;
        val rev = new int[n];
        for (var i = 0; i < n; i++) rev[i] = original[n - 1 - i];
        return rev;
    }

    public NArray flattened() { return new NArray(size).fill(buffer()); }

    public NArray reshaped(int... shape) { return new NArray(shape).fill(buffer()); }

    public NArray map(@NonNull DoubleUnaryOperator op) {
        val arr = new NArray(shape);
        IntStream.range(0, size).parallel().forEach(i -> arr[i] = op.applyAsDouble(buffer[i]));
        return arr;
    }

    public NArray map(@NonNull DoubleUnaryOperator... ops) {
        val arr = new NArray(shape);
        val op = chain(ops);
        IntStream.range(0, size).parallel().forEach(i -> arr[i] = op.applyAsDouble(buffer[i]));
        return arr;
    }

    private DoubleUnaryOperator chain(DoubleUnaryOperator... ops) {
        int n = ops.length;
        if (n == 0) return DoubleUnaryOperator.identity();
        var op = ops[0];
        for (var i = 1; i < n; i++) op = op.andThen(ops[i]);
        return op;
    }

    public NArray bimap(@NonNull DoubleBinaryOperator op, @NonNull NArray arg2) {
        if (arg2.size != size) {
            val arr = new NArray(broadcast(shape, arg2.shape));
            arr.coordinates().parallel().forEach(
                coords -> { arr[coords] = op.applyAsDouble(this[clipCoords(coords)], arg2[arg2.clipCoords(coords)]); });
            return arr;
        }
        val arr = Arrays.equals(arg2.shape, shape) ? new NArray(shape) : new NArray(new int[] { size });
        IntStream.range(0, size).parallel().forEach(i -> arr[i] = op.applyAsDouble(this[i], arg2[i]));
        return arr;
    }

    public NArray bimap(@NonNull DoubleBinaryOperator op, double arg2) {
        val arr = new NArray(shape);
        IntStream.range(0, size).parallel().forEach(i -> arr[i] = op.applyAsDouble(this[i], arg2));
        return arr;
    }

    public NArray bimapRev(@NonNull DoubleBinaryOperator op, double arg2) {
        val arr = new NArray(shape);
        IntStream.range(0, size).parallel().forEach(i -> arr[i] = op.applyAsDouble(arg2, this[i]));
        return arr;
    }

    public int[] clipCoords(@NonNull int... coords) {
        val N = coords.length;
        if (N > dims) { return Arrays.copyOfRange(coords, N - dims, N); }
        return coords;
    }

    public static int[] broadcast(@NonNull int[]... shapes) {
        if (shapes.length < 2) throw new IllegalArgumentException(
                "Atleast 2 shapes are needed for broadcasting. Found " + Arrays.deepToString(shapes));
        val N = Stream.of(shapes).mapToInt(shape -> shape.length).max().getAsInt();
        val bshape = new int[N];
        for (val shape : shapes) {
            val dim = shape.length;
            for (int i = 0; i < dim; i++) {
                int j = i + N - dim;
                val b = bshape[j];
                val s = shape[i];
                if (s < 1) throw new IllegalArgumentException(
                        "Illegal shape. Wrong cardinality provided { < 1 }. " + Arrays.deepToString(shapes));
                if (s > 1 && b > 1 && b != s)
                    throw new IllegalArgumentException("Shapes can not be broadcast. " + Arrays.deepToString(shapes));
                if (b < 2 && b != s) bshape[j] = s;
            }
        }
        return bshape;
    }

    public double reduce(double identity, DoubleBinaryOperator op) { return values().reduce(identity, op); }

    public NArray reduceDim(int dim, double identity, DoubleBinaryOperator op) {
        Objects.checkIndex(dim, dims);
        if (dim == 0) return reduceFirst(identity, op);
        if (dim == this.dims - 1) return reduceLast(identity, op);
        val dims = this.dims - 1;
        val shape = new int[dims];
        arraycopy(this.shape, 0, shape, 0, dim);
        arraycopy(this.shape, dim + 1, shape, dim, dims - dim);
        val arr = new NArray(shape);
        val dsize = this.shape[dim];
        arr.coordinates().forEach(coords -> {
            val pcoords = new int[this.dims];
            arraycopy(coords, 0, pcoords, 0, dim);
            arraycopy(coords, dim, pcoords, dim + 1, dims - dim);
            arr[coords] = IntStream.range(0, dsize).parallel().mapToDouble(i -> {
                val tcoords = pcoords.clone();
                tcoords[dim] = i;
                return this[tcoords];
            }).reduce(identity, op);
        });
        return arr;
    }

    public NArray reduceDim(int dimFrom, int dimTo, double identity, DoubleBinaryOperator op) {
        Objects.checkFromToIndex(dimFrom, dimTo, this.dims);
        if (dimFrom == dimTo) return reduceDim(dimTo, identity, op);
        if (dimFrom == 0 && dimTo == this.dims - 1) return NArray.of(reduce(identity, op));
        val dims = this.dims - (dimTo + 1 - dimFrom);
        val shape = new int[dims];
        arraycopy(this.shape, 0, shape, 0, dimFrom);
        arraycopy(this.shape, dimTo + 1, shape, dimFrom, dims - dimFrom);
        val arr = new NArray(shape);
        this.coordinates().collect(Collectors.groupingBy(coords -> {
            val dcoords = new int[dims];
            arraycopy(coords, 0, dcoords, 0, dimFrom);
            arraycopy(coords, dimTo + 1, dcoords, dimFrom, dims - dimFrom);
            return comparable.of(dcoords);
        })).forEach((d, s) -> arr[d.arr] = s.parallelStream().mapToDouble(this::get).reduce(identity, op));
        return arr;
    }

    public NArray reduceFirst(double identity, DoubleBinaryOperator op) {
        if (dims == 1) return NArray.of(reduce(identity, op));
        val arr = new NArray(Arrays.copyOfRange(shape, 1, dims));
        val dfirst = shape[0];
        val firstBs = blockSize[0];
        IntStream.range(0, arr.size).forEach(i -> arr[i] = IntStream.range(0, dfirst).parallel()
                .mapToDouble(j -> buffer[i + j * firstBs]).reduce(identity, op));
        return arr;
    }

    public NArray reduceFirstDims(int dims, double identity, DoubleBinaryOperator op) {
        if (dims <= 0 || dims > this.dims) throw new IllegalArgumentException();
        if (dims == 1) return reduceFirst(identity, op);
        if (dims == this.dims) return NArray.of(reduce(identity, op));
        val arr = new NArray(Arrays.copyOfRange(shape, dims, this.dims));
        val firstBs = blockSize[dims - 1];
        IntStream.range(0, arr.size).forEach(i -> arr[i] = IntStream.range(0, size / firstBs).parallel()
                .mapToDouble(j -> buffer[i + j * firstBs]).reduce(identity, op));
        return arr;
    }

    public NArray reduceLast(double identity, DoubleBinaryOperator op) {
        if (dims == 1) return NArray.of(reduce(identity, op));
        val arr = new NArray(Arrays.copyOf(shape, dims - 1));
        val lastBs = shape[dims - 1];
        IntStream.range(0, arr.size).forEach(i -> arr[i] = IntStream.range(0, lastBs).parallel()
                .mapToDouble(j -> buffer[i * lastBs + j]).reduce(identity, op));
        return arr;
    }

    public NArray reduceLastDims(int dims, double identity, DoubleBinaryOperator op) {
        if (dims <= 0 || dims > this.dims) throw new IllegalArgumentException();
        if (dims == 1) return reduceLast(identity, op);
        if (dims == this.dims) return NArray.of(reduce(identity, op));
        val arr = new NArray(Arrays.copyOf(shape, this.dims - dims));
        val lastBs = blockSize[dims - 2];
        IntStream.range(0, arr.size).forEach(i -> arr[i] = IntStream.range(0, lastBs).parallel()
                .mapToDouble(j -> buffer[i * lastBs + j]).reduce(identity, op));
        return arr;
    }

    public int[] getShape() { return shape.clone(); }

    public int dim(int index) { return shape[index]; }

    public int firstDim() { return shape[0]; }

    public int lastDim() { return shape[dims - 1]; }

    public Stream<int[]> coordinates() { return IntStream.range(0, size).mapToObj(this::coordinatesOf); }

    public DoubleStream values() { return IntStream.range(0, size).mapToDouble(this::get); }

    public DoubleBuffer buffer() { return buffer.asReadOnlyBuffer().rewind(); }

    @Override
    public Iterator<Double> iterator() { return values().iterator(); }

    @Override
    public NArray clone() { return new NArray(shape).fill(buffer); }

    @Override
    public String toString() { return toPrettyString(10); }

    public String toPrettyString(int limit, @NonNull int... coords) {
        int N = coords.length;
        if (N > dims) throw new IllegalArgumentException(
                "Coordinates dimensionality " + N + ", is out of bounds for " + getClass().getSimpleName() + shape);
        if (N == dims) return Double.toString(this[coords]);
        else {
            int n = shape[N];
            val list = IntStream.range(0, Math.min(limit, n)).mapToObj(i -> {
                val newCoord = Arrays.copyOf(coords, N + 1);
                newCoord[N] = i;
                return toPrettyString(limit, newCoord);
            }).collect(Collectors.toList());
            if (n > limit) { list.add(String.format("... %d more", n - limit)); }
            if (N == dims - 1) {
                val str = list.stream()//
                        .reduce((a, b) -> a + ", " + b)//
                        .orElse("");
                return "[ " + str + " ]";
            } else {
                val str = list.stream()//
                        .map(s -> "\t" + s.toString().replaceAll("\n", "\n\t"))//
                        .reduce((a, b) -> a + "\n" + b)//
                        .orElse("");
                return "[\n" + str + "\n]";
            }
        }
    }

    @Value
    @AllArgsConstructor(staticName = "of")
    private static class comparable {
        int[] arr;
    }

    public static void main(String[] args) {
        val arr = new NArray(new int[] { 3, 4 }, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        val arr2 = arr.T();
        val arr3 = arr.dot(arr2);
        val arr4 = arr3.reduceLast(0, Double::sum);
        val arr5 = arr.reshaped(3, 2, 2);
        val arr6 = arr5.reduceFirst(0, Double::sum);
        println(arr);
        println(arr2);
        println(arr3);
        println(arr4);
        println(arr5);
        println(arr5.reduceLastDims(2, 0, Double::sum));
        println(arr6);
        println(arr5.reduceFirstDims(2, 0, Double::sum));
        println(arr5.reduceDim(1, 0, Double::sum));
        println(arr5.reduceDim(0, 1, 0, Double::sum));
        println(arr.map(Math::sqrt, Math::sin));
        println(Arrays.toString(broadcast(new int[] { 3, 2, 4 }, new int[] { 3, 2, 1 })));

        println(of(arr, arr, arr));
    }

}
