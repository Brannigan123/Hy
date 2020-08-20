package hy.layer.feedforward;

import static java.lang.Math.abs;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;
import static java.lang.Math.min;
import static java.lang.Math.pow;

import hy.util.NArray;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class Activation {

    public final FeedForwardLayer linear      = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input; }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) { return delta; }

                                                  @Override
                                                  public String toString() { return "Linear Activation"; }
                                              };

    public final FeedForwardLayer sigmoid     = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> 1.0 / (1.0 + exp(-x))); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(y -> y * (1.0 - y)) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "Sigmoid Activation"; }
                                              };

    public final FeedForwardLayer hardSigmoid = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input, boolean isTraining) {
                                                      return input
                                                              .map(x -> min(max(x * 0.2 + 0.5, 0.0), 1.0));
                                                  }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(y -> y > 0.0 && y < 1.0 ? 0.2 : 0.0)
                                                             * delta;
                                                  }

                                                  @Override
                                                  public String
                                                      toString() { return "Hard Sigmoid Activation"; }
                                              };

    public final FeedForwardLayer tanh        = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input.map(Math::tanh); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.bimap((y, dy) -> (1.0 - y * y) * dy,
                                                          delta);
                                                  }

                                                  @Override
                                                  public String toString() { return "Tanh Activation"; }
                                              };

    public final FeedForwardLayer gauss       = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> exp(-x * x)); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return input.bimap((x, y) -> -2.0 * x * y, output)
                                                             * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "Tanh Activation"; }
                                              };

    public final FeedForwardLayer relu        = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> max(0.0, x)); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(y -> y > 0.0 ? 1.0 : 0.0) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "ReLU Activation"; }
                                              };

    public final FeedForwardLayer relu6       = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> min(max(0.0, x), 6.0)); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(
                                                          y -> (y > 0.0) && (y < 6.0) ? 1.0 : 0.0) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "ReLu6 Activation"; }
                                              };

    public final FeedForwardLayer relu3       = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> min(max(0.0, x), 3.0)); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(
                                                          y -> (y > 0.0) && (y < 3.0) ? 1.0 : 0.0) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "ReLU3 Activation"; }
                                              };

    public final FeedForwardLayer lrelu       = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input,
                                                      boolean isTraining) { return input
                                                              .map(x -> x > 0.0 ? x : x * 0.01); }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(y -> y > 0.0 ? 1.0 : 0.01) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "LReLU Activation"; }
                                              };

    public final FeedForwardLayer elu         = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input, boolean isTraining) {
                                                      return input.map(x -> max(1.0 * (exp(x) - 1.0), x));
                                                  }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(y -> y > 0 ? 1.0 : 1.0 * exp(y))
                                                             * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "ELU Activation"; }
                                              };

    public final FeedForwardLayer selu        = new FeedForwardLayer() {
                                                  double alpha = 1.6732632423543772848170429916717;
                                                  double scale = 1.0507009873554804934193349852946;

                                                  @Override
                                                  public NArray of(NArray input, boolean isTraining) {
                                                      return input.map(
                                                          x -> max(scale * alpha * (exp(x) - 1.0), x));
                                                  }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      return output.map(
                                                          y -> y > 0 ? 1.0 : scale * alpha * exp(y)) * delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "SELU Activation"; }
                                              };

    public final FeedForwardLayer softmax     = new FeedForwardLayer() {
                                                  @Override
                                                  public NArray of(NArray input, boolean isTraining) {
                                                      val max = input.reduceLast(0, Math::max);
                                                      val exps = input.bimap((x, m) -> exp(x - m), max);
                                                      val sums = exps.reduceLast(0, Double::sum);
                                                      return exps / sums;
                                                  }

                                                  @Override
                                                  public NArray delta(NArray input, NArray output,
                                                      NArray delta) {
                                                      // if loss cross entropy
                                                      return delta;
                                                  }

                                                  @Override
                                                  public String toString() { return "Softmax Activation"; }
                                              };

    public FeedForwardLayer lrelu(double alpha) {
        return new FeedForwardLayer() {
            @Override
            public NArray of(NArray input, boolean isTraining) {
                return input.map(x -> x > 0.0 ? x : x * alpha);
            }

            @Override
            public NArray delta(NArray input, NArray output, NArray delta) {
                return output.map(y -> y > 0.0 ? 1.0 : alpha) * delta;
            }

            @Override
            public String toString() { return "LReLU (alpha=" + alpha + ") Activation"; }
        };
    }

    public FeedForwardLayer elu(double alpha) {
        return new FeedForwardLayer() {
            @Override
            public NArray of(NArray input, boolean isTraining) {
                return input.map(x -> max(alpha * (exp(x) - 1.0), x));
            }

            @Override
            public NArray delta(NArray input, NArray output, NArray delta) {
                return output.map(y -> y > 0 ? 1.0 : alpha * exp(y)) * delta;
            }

            @Override
            public String toString() { return "ELU (alpha=" + alpha + ") Activation"; }
        };
    }

    public FeedForwardLayer radial(double altitude) {
        return new FeedForwardLayer() {
            @Override
            public NArray of(NArray input, boolean isTraining) {
                return input.map(x -> exp(input.size * log(altitude) - x));
            }

            @Override
            public NArray delta(NArray input, NArray output, NArray delta) {
                return input.bimap((x, dy) -> (-pow(altitude, input.size) / exp(x)) * dy, delta);
            }

            @Override
            public String toString() { return "Radial Basic (altitude=" + altitude + ") Activation"; }
        };
    }

    public FeedForwardLayer softplus(double beta) {
        return new FeedForwardLayer() {
            @Override
            public NArray of(NArray input, boolean isTraining) {
                input = input * beta;
                val max = input.reduceLast(0, Math::max);
                return (max + input.map(x -> log(1.0 + exp(-abs(x))))) / beta;
            }

            @Override
            public NArray delta(NArray input, NArray output, NArray delta) {
                return output.bimap((y, dy) -> (1.0 - 1.0 / (1.0 + exp(beta * y))) * dy, delta);
            }

            @Override
            public String toString() { return "Softplus (beta=" + beta + ") Activation"; }
        };
    }

}
