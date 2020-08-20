package hy.loss;

import hy.util.NArray;
import lombok.val;

public interface Loss {
    NArray of(NArray prediction, NArray target);

    NArray gradient(NArray prediction, NArray target);

    public final Loss square              = new Loss() {
                                              @Override
                                              public NArray of(NArray prediction, NArray target) {
                                                  val n = prediction.size;
                                                  return prediction.bimap((p, t) -> Math.pow(p - t, 2) / n,
                                                      target);
                                              }

                                              @Override
                                              public NArray gradient(NArray prediction, NArray target) {
                                                  return prediction.bimap((p, t) -> 2 * (p - t), target);
                                              }

                                              @Override
                                              public String toString() { return "Square Loss"; }
                                          };

    public final Loss logCosh             = new Loss() {
                                              @Override
                                              public NArray of(NArray prediction, NArray target) {
                                                  val n = prediction.size;
                                                  return prediction.bimap(
                                                      (p, t) -> Math.log(Math.cosh(p - t)) / n, target);
                                              }

                                              @Override
                                              public NArray gradient(NArray prediction, NArray target) {
                                                  return prediction.bimap((p, t) -> Math.tanh(p - t), target);
                                              }

                                              @Override
                                              public String toString() { return "Log Cosh Loss"; }
                                          };

    public final Loss softmaxCrossEntropy = new Loss() {
                                              @Override
                                              public NArray of(NArray prediction, NArray target) {
                                                  val n = prediction.size;
                                                  return prediction.bimap((p, t) -> t * -Math.log(p) / n,
                                                      target);
                                              }

                                              @Override
                                              public NArray gradient(NArray prediction, NArray target) {
                                                  // if output layer is softmax
                                                  return prediction - target;
                                              }

                                              @Override
                                              public String toString() { return "Softax CE Loss"; }
                                          };
    public final Loss binaryCrossEntropy  = new Loss() {
                                              @Override
                                              public NArray of(NArray prediction, NArray target) {
                                                  val n = prediction.size;
                                                  return prediction
                                                          .bimap(
                                                              (p, t) -> (t * -Math.log(p)
                                                                  + (1.0 - t) * -Math.log(1.0 - p)) / n,
                                                              target);
                                              }

                                              @Override
                                              public NArray gradient(NArray prediction, NArray target) {
                                                  return prediction.bimap((p, t) -> (p - t) / (p * (1 - p)),
                                                      target);
                                              }

                                              @Override
                                              public String toString() { return "Binary CE Loss"; }
                                          };
}
