# Hypatia (Hy) Neural Net Library
Simple Neural Network implementation from scratch, meant to solidify my understanding of the topic.

## Sample Screenshots [link to this sample](https://github.com/Brannigan123/FaceID)
### Training
![Training Graph](https://github.com/Brannigan123/Hy/blob/master/images/Training.PNG)
### Inference
![Alice image 1 inference](https://github.com/Brannigan123/Hy/blob/master/images/Alice.PNG)
![Alice image 2 inference](https://github.com/Brannigan123/Hy/blob/master/images/Alice%202.PNG)
![Carlos image 1 inference](https://github.com/Brannigan123/Hy/blob/master/images/Carlos%202.PNG)
![Carlos image 2 inference](https://github.com/Brannigan123/Hy/blob/master/images/Carlos%203.PNG)
## Snippet
### Xor Gate
Here is the classic xor gate implemented with a 3 layer FC neural net.

```java
// imports
import static hy.API.*;
import lombok.val;

```

```java
// Defining the architecture
val nn = Sequential()
         .add(FC(2, 3), Tanh)
         .add(FC(3, 2), Tanh)
         .add(FC(2, 1));

// Defining the training data
val x = List.of(
          Array(0, 0),
          Array(0, 1),
          Array(1, 0),
          Array(1, 1)
        );

val y = List.of(Array(0), Array(1), Array(1), Array(0));

// Defining the training configuration
val trainConfig = TrainConfig()
          .inputs(x)
          .targets(y)
          .batch(2)
          .shuffle(true)
          .minLoss(1e-3)
          .epochs(2000)
          .build();

// Train
nn.fit(trainConfig);

// Log final results
println(nn.predict(Array(0, 0)));
println(nn.predict(Array(0, 1)));
println(nn.predict(Array(1, 0)));
println(nn.predict(Array(1, 1)));

```
Sample xor training log.

```
epoch    1/2000 Log Cosh Loss = 0.3441278070029366
epoch    2/2000 Log Cosh Loss = 0.24375407153465225
...
epoch  865/2000 Log Cosh Loss = 0.001045385944558071
epoch  866/2000 Log Cosh Loss = 0.001005062165002299
epoch  867/2000 Log Cosh Loss = 0.0010004123525764889
██████████████████████████                         Average Log Cosh Loss =   0.0008173355299494, 1/2 of Batch 1

```
Sample xor result

```
[ 0.047311548915308754 ]
[ 0.9551353389694562 ]
[ 0.9596545718172823 ]
[ 0.043970193115603906 ]
```

## Features
- Sequential Model
- Layers
	- Feed-forward layers
		- Fully connected layer
		- Convolution layers (1d-3d)
		- Transposed Convolution layers (1d-3d)
		- Sampling layer
		- Pooling layers (max/avg)
		- Dropout layer
		- Scaling layer
		- Flatten layer
		- Reshape layer
	- Recurrent layers (cells) [TODO]
		- RNN
		- GRU
		- LSTM
	- Activation functions
		- Linear
		- Sigmoid
		- Hard Sigmoid
		- Tanh
		- Gaussian
		- ReLU, ReLU3 & ReLU6
		- LReLU
		- ELU
		- SELU
		- Softplus
		- Softmax
- Loss functions
	Square loss, LogCosh loss, Binary CrossEntropy & Softmax CrossEntropy
- Optimizers
	SGD, Momentum SGD, NAG, RMSProp, AdaGrad & Adam
- Weight Regularizer
	L1, L2 & Elastic Net Regularizers

Helper functions are available in [hy.API](https://github.com/Brannigan123/Hy/blob/master/src/hy/API.java) or alternatively make use of builder methods of respective classes.

## More Samples
### Convolutional Neural Network
```java
Sequential featureExtractor = Sequential(        // 3*256*256
          .add(Conv(4, 3, 3))                    // 4*254*254
          .add(Conv(5, 4, 3))                    // 5*252*252
          .add(MaxPool(2), Gauss)                // 5*126*126
          .dropout(0.3)                          // 5*126*126
          .add(Conv(6, 5, 3))                    // 6*124*124
          .add(Conv(6, 6, 3))                    // 6*122*122
          .add(MaxPool(2), Gauss)                // 6*61*61
          .dropout(0.3)                          // 6*61*61
          .add(Conv(5, 6, 3))                    // 5*59*59
          .add(Conv(5, 5, 3))                    // 5*57*57
          .add(MaxPool(3), Gauss)                // 5*19*19
          .load(encoderParamPath)                //
;

Sequential classifier = Sequential()             // 5*19*19
          .flatten()                             // 1805
          .add(FC(1805, 1000), Tanh)             // 1000
          .dropout(0.3)                          // 1000
          .add(FC(1000, 500), Tanh)              // 500
          .dropout(0.2)                          // 500
          .add(FC(500, 50), Tanh)                // 50
          .add(FC(50, 20), Tanh)                 // 20
          .add(FC(20, numOfClasses), Softmax)    // # classes
          .load(fcParamPath)                     //
;

Sequential cnn = Sequential()                    // 3*256*256
          .add(featureExtractor)                 // 5*19*19
          .add(classifier)                       // # classes
;

 val trainConfig = TrainConfig()
          .inputs(images)
          .targets(labels)
          .batchSize(20)
          .shuffle(true)
          .loss(SoftmaxCrossEntropy)
          .minLoss(1e-3)
          .regularizer(L2Regularizer(0.006))
          .epochs(500)
          .build();

cnn.fit(trainConfig);

```

### Fully Convolutional AutoEncoder
```java
Sequential encoder = Sequential()                // 3*256*256
          .add(Conv(4, 3, 3))                    // 4*254*254
          .add(Conv(5, 4, 3))                    // 5*252*252
          .add(MaxPool(2), Gauss)                // 5*126*126
          .dropout(0.3)                          // 5*126*126
          .add(Conv(6, 5, 3))                    // 6*124*124
          .add(Conv(6, 6, 3))                    // 6*122*122
          .add(MaxPool(2), Gauss)                // 6*61*61
          .dropout(0.3)                          // 6*61*61
          .add(Conv(5, 6, 3))                    // 5*59*59
          .add(Conv(5, 5, 3))                    // 5*57*57
          .add(MaxPool(3), Gauss)                // 5*19*19
          .load(encoderParamPath)                //
;

Sequential decoder = Sequential()                // 5*19*19
          .add(TConv(5, 5, 3, 3, 0))             // 5*57*57
          .add(TConv(5, 5, 3), Gauss)            // 5*59*59
          .dropout(0.3)                          // 5*59*59
          .add(TConv(6, 5, 3))                   // 6*61*61
          .add(TConv(6, 6, 4, 2, 0), Gauss)      // 6*124*124
          .dropout(0.3)                          // 6*124*124
          .add(TConv(5, 6, 3))                   // 5*126*126
          .add(TConv(4, 5, 3, 2, 0), Gauss)      // 5*253*253
          .add(TConv(3, 4, 4, 1, 0))             // 3*256*256
          .load(decoderParamPath)                //
;

Sequential ae = Sequential()                     // 3*256*256
          .add(encoder)                          // 5*19*19
          .add(decoder)                          // 3*256*256
;

 val trainConfig = TrainConfig()
          .inputs(x)
          .targets(x)
          .batchSize(4)
          .shuffle(true)
          .loss(SquareLoss)
          .minLoss(1e-3)
          .regularizer(L2Regularizer(0.006))
          .epochs(500)
          .build();

ae.fit(trainConfig);

```