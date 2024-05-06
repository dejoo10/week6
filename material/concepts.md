# Convolutional Neural Network (CNN)

<img src="./img/Convolution-Max-Pooling-Flatten.jpg" alt="" width=auto>

```python
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(472, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
```


## Steps

- Image channels
- Convolution
- Pooling
- Flattening
- Full connection

### 1. Image channels

- Demo: [image](./img/apple.jpg) to [RGB](https://onlinetools.com/image/separate-image-color-channels)


<img src="./img/v1.png" alt="" width="50%">

<img src="./img/1a.png" alt="" width=auto>

<img src="./img/1b.jpg" alt="" width=auto>


### 2. Convolution

**Feature maps**

<img src="./img/2a1.png" alt="" width=auto>

<img src="./img/2a2.gif" alt="" width=auto>

<img src="./img/2b1.jpg" alt="" width=auto>

<img src="./img/2b2.png" alt="" width=auto>

**Kernels**

<img src="./img/2c.jpg" alt="" width=auto>


**Striding**

<img src="./img/2d.gif" alt="" width=auto>

**Padding**

<img src="./img/2e.gif" alt="" width=auto>

**Convolutions applied over the RGB channels**

<img src="./img/2f.png" alt="" width=auto>

**Convolutions applied to more than one filter**

<img src="./img/2g.png" alt="" width=auto>

### 3. Pooling: min, max, and average pooling

**Max pooling applied to a filter of size 2 (2x2) and stride=1**

<img src="./img/3a.gif" alt="" width=auto>

**Another example**

<img src="./img/3b.jpg" alt="" width=auto>

### 4. Flattening

<img src="./img/4a.gif" alt="" width=auto>

### 5. Full connection

<img src="./img/5a.png" alt="" width=auto>

### 6. CNN Architecture

<img src="./img/cnn-architecture.jpg" alt="" width=auto>

**Types of convolutional neural networks**

- AlexNet
- VGGNet
- GoogLeNet
- ResNet
- ZFNet
<!-- <img src="./img" alt="" width=auto> -->

-------
## Ref

- [Introduction to convolutional neural networks](https://developer.ibm.com/articles/introduction-to-convolutional-neural-networks/?mhsrc=ibmsearch_a&mhq=convolutional%20neural%20networks%26quest%3B)
- Other:
  - [How do convolutional neural networks work?](https://www.ibm.com/topics/convolutional-neural-networks)
  - Books:
    - Deep Learning with Python, Second Edition, By Francois Chollet
    - Deep Learning from Scratch, By Seth Weidman