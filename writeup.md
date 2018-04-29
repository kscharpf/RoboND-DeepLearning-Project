#Udacity Robotics Term 1 Deep Learning Project
## Neural Network Architecture
This specification for this project is a fully convolutional network (FCN) with some number of encoding and decoding layers. The encoding layers are separable convolutional blocks. 

### High Level Architecture
The high-level architecture of the built FCN looks like

| Layer | Filters | Output Shape |
| -------- |:----------:|:------------------:|
| Input | N/A  | 160x160x3  |
| Encoder 1 | 64 | 80x80x64 |
| Encoder 2 | 256 | 40x40x256 |
| Encoder 3 | 512 | 20x20x512 |
| 1x1 Convolution | 512 | 20x20x512 |
| Decoder 1 | 512 | 40x40x512 | 
| Decoder 2 | 256 | 80x80x256 |
| Decoder 3 | 64 | 160x160x64 |
| Conv2D Output | 3 | 160x160x3 |

The depth of 3 for the input layer corresponds to the RGB color scheme - 1 value for each of the color components. The depth of 3 for the output layer corresponds to the number of classes we recognize in this neural network - target human, other human, everything else.

The number of encoder and decoder layers as well as the number of filters chosen within each layer was partially determined by the memory limitations of the machine used for training. It was not possible in my environment to exceed 512 filters. Nor was it possible to exceed three encoder / decoder layers given the use of two layers within each encoding / decoding layer as noted below. In general, I tried to maximize the number of filters available given the memory constraints. 

The 1x1 convolutional layer in this case serves to add nonlinearity to the learned function as the number of filters is the same as the input. 

[image_0]: ./fcnarch.png
![Figure 1: Fully Convolutional Network Architecture][image_0] 

### Encoder Architecture
The encoder layer itself consists of two separable 2D convolutional batch normalization layers. The second layer conv2d layer uses a stride of 1 so as not to impact the shape. Experimentation with architectures revealed that this additional stride 1 layer was valuable in producing a better IoU. The convolutional layer uses a kernel size of 3 which is a standard recommendation employed in VGG16 among others.

| Layer | Stride |
| ------- |:--------:|
| Conv2D | 2 |
| Batch Normalization | N/A |
| Conv2D | 1 |
| Batch Normalization | N/A |

### Decoder Architecture
The decoder layer combines the inputs from the previous layer with the outputs of an earlier layer and runs two separable 2D convolutional batch normalization layers on the concatenated result. In general, a decoder layer takes as input the output of the previous layer having shape (x,x,w) and the output of a decoder layer of shape (2x,2x,y). In order to combine these two layers, the (x,x,w) layer is upsampled to (2x,2x,w) resulting in the concatenated layer (2x,2x,w+y). It is on that concatenated result that we run the convolution producing an output of (2x,2x,z) where z is the number of filters in the convolutional layer.

| Layer | Stride |
| ------- |:--------:|
| Upsample | |
| Concatenate ||
| Conv2D | 1 |
| Batch Normalization | |
| Conv2D | 1 |
| Batch Normalization | |

## Training Data Collection
I constructed a variety of scenes within the simulation with the target hero, non-target humans, varying backgrounds, and varying elevation of the quad. I generally tried to collect at least 500 images from each camera for a given scene, sometimes as many as several thousand from each camera. From this raw data set, I downsampled to force mostly images with the target hero. For example, given a data set I would target that 80% of the resulting images had the hero. The remaining 20% of images would be randomly selected from the other images. After preprocessing, I used the sklearn train_test_split function to randomly allocate 80% of these images to training and 20% to validation. The end result after preprocessing was that I added ~4800 images to the training data set and ~1300 images to the validation data set. 

## Hyperparameter Selection
The overall method of hyperparameter selection was brute force and the end result was:

| Parameter | Value |
| ------------- |:--------:|
| Epochs | 15 |
| Batch Size | 40 |
| Learning Rate | 0.0002 |
| Training Steps per Epoch | 225 |
| Validation Steps per Epoch | 61 |

The batch size was chosen based upon memory limitations of the AWS instance. The number of training steps per epoch and validation steps per epoch were derived from the total number of samples and the batch size.

	training_steps = ceil(training_samples / batch_size)
	validation_steps = ceil(validation_samples / batch_size)
	
The learning rate was chosen as a result of experimentation. Values of 0.001 and 0.0005 seemed to result in convergence difficulties while 0.0001 simply converged too slowly resulting in too long training stages. 0.0002 with the number of epochs set to 15 avoided overfitting while keeping the total training time on the AWS instance less than 2.5h.

Figure 2 shows the training and validation loss curve.

[image_1]: ./losscurve.png  
![Figure 2: Training and Validation Loss Versus Epoch][image_1]

The table below shows the specific training and validation losses for each training epoch. As you can see, the training and validation losses fall sharply for the first several epochs while stabilizing after epoch 10.

| Epoch | Training Loss | Validation Loss | 
| ------- |:----------------:|:-------------------:|
| 1 | 0.839 | 0.557 |
| 2 | 0.335 | 0.175 |
| 3 | 0.144 | 0.093 |
| 4 | 0.080 | 0.053 | 
| 5 | 0.054 | 0.047 |
| 6 | 0.041 | 0.039 |
| 7 | 0.035 | 0.033 |
| 8 | 0.029 | 0.028 | 
| 9 | 0.025 | 0.024 |
| 10 | 0.023 | 0.025 | 
| 11 | 0.021 | 0.024 |
| 12 | 0.019 | 0.023 | 
| 13 | 0.018 | 0.024 |
| 14 | 0.017 | 0.024 |
| 15 | 0.016 | 0.021 |

## Results
IoU and evaluation results are shown in the table below

| Test Name | True Positives | False Positives | False Negatives | IoU Background | IoU Other People | IoU Hero |
| ------------- |:-----------------:|:--------------:|:---------------:|:--------------:|:--------------:|:-----------------:|
| Following | 539 | 0 | 0 | 0.996 | 0.360 | 0.887 |
| Non-Visible | 0 | 18 | 0 | 0.989 | 0.781 | 0.0 |
| Visible | 131 | 2 | 170 | 0.997 | 0.441 | 0.245 |

The final score calculated from these intermediate results was** 0.441**.

## Experimentation
The quadcopter successfully found the hero however the inference engine appears to have run too slowly on my host (no gpu). The result was stalls in the quad and the textual output of the inference processing reporting the target found followed by target lost.

## Limitations
The developed model is only valid for differentiating three object types - target human, other human, and everything else. It seems that it should be possible to use a pre-trained inception/resnet/vgg16 model on the CIFAR1000 dataset and use that as part of the segmentation process. 






