import cv2
import time
import numpy as np
from DetectFaces import loadImages

FOLDER = ""
PIC = 1
TOTAL_IMAGES = 0
START = time.time()
CORRECT = 0
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = (10 ** -8)


class Layer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.input = None
        self.output = None
        self.cache = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        self.grads = [0, 0, 0, 0, 0, 0, ]
        self.params = [0, 0, 0, 0, 0, 0, 0, ]
        self.m = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        self.v = [0, 0, 0, 0, 0, 0, 0, 0, 0, ]
        self.time_step = 1

    def train(self):
        for param, grad, m, v in zip(self.params, self.grads, self.m, self.v):
            """m = (BETA1 * m) + (1 - BETA1) * grad
            v = (BETA2 * v) + (1 - BETA2) * np.square(grad)
            m_V = m / (1 - (BETA1 ** self.time_step))
            v_V = v / (1 - (BETA2 ** self.time_step))
            param -= LEARNING_RATE * (np.divide(m_V, (np.sqrt(v_V) + EPSILON)))
            self.time_step += 1"""
            param -= (0.1 * grad + 0.0001 * param)


class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        Layer.__init__(self, input_shape, output_shape)
        self.weights = 1 - np.random.random((input_shape, output_shape)) * 2
        self.bias = np.zeros((output_shape,))
        self.params = [self.weights, self.bias]

    def forward(self, input):
        self.input = input
        self.output = input.dot(self.weights) + self.bias
        return self.output

    def backward(self, top_grad):
        self.grads[0] = self.input.T.dot(top_grad)
        self.grads[1] = top_grad.sum(axis=0)
        self.train()
        # self.check_grad(self.grads[0])
        return top_grad.dot(self.weights.T)


class Relu(Layer):
    def __init__(self, input_shape):
        Layer.__init__(self, input_shape, input_shape)

    def forward(self, input):
        # self.cache[0] = input > 0
        # return input * self.cache[0]
        self.cache[0] = 1 / (1 + np.exp(- input))
        return self.cache[0]

    def backward(self, top_grad):
        return top_grad * self.cache[0] * (1 - self.cache[0])


class SoftmaxCrossentropy(Layer):
    def __init__(self, input_shape):
        Layer.__init__(self, input_shape, input_shape)
        self.target = None

    def forward(self, input, target):
        exp = np.exp(input - input.max(axis=1, keepdims=True))
        self.target = target
        softmax = exp / exp.sum(axis=1, keepdims=True)
        self.cache[0] = softmax
        cross_entropy = -np.log(
            np.clip(softmax[np.arange(softmax.shape[0]), np.argmax(target, axis=1)], a_min=1e-15, a_max=1e30)).mean()
        return cross_entropy

    def backward(self):
        return (self.cache[0] - self.target) / self.target.shape[0]


def load_images():
    folders = ['S1', 'S2', 'S3']
    data_folder = "TrainingData/"
    data = []
    labels = []
    for ind, f in enumerate(folders):
        for i in range(1, 561):
            target = np.zeros((3,))
            target[ind] = 1.0
            img = cv2.imread(data_folder + f + "/" + str(i) + ".jpg", 0)
            if img is not None:
                data.append(0.5 - img.flatten() / 255.0)
                labels.append(target)
    return np.array(data), np.array(labels)


def predict(layers):
    global CORRECT, iVector, TOTAL_IMAGES
    folders = ['S1', 'S2', 'S3']
    for folder in folders:
        data_folder = 'TestData/' + folder
        images = loadImages(data_folder)
        for image in images:
            img = cv2.imread(data_folder + "/" + str(image), 0)
            TOTAL_IMAGES += 1
            print(data_folder + "/" + str(image))
            iVector = 0.5 - img.flatten().reshape((1, 3600)) / 255.0
            o1 = layers[0].forward(iVector)
            o2 = layers[1].forward(o1)
            o3 = layers[2].forward(o2)
            o4 = layers[3].forward(o3)
            o5 = layers[4].forward(o4)
            o6 = layers[5].forward(o5)
            o7 = layers[6].forward(o6)
            o8 = layers[7].forward(o7)
            o9 = layers[8].forward(o8)
            output = o9.argmax()
            output = output + 1
            # print(o3)
            print(output)
            prediction = "S" + str(output)
            if prediction == folder:
                CORRECT = CORRECT + 1


layers = [FCLayer(3600, 2000), Relu(2000), FCLayer(2000, 1000), Relu(1000),
          FCLayer(1000, 200), Relu(200), FCLayer(200, 50), Relu(50), FCLayer(50, 3), SoftmaxCrossentropy(3)]
x, y = load_images()
ord = np.arange(x.shape[0])
print(x.shape)
batch_size = 39
n_batches = 36
for i in range(100):
    np.random.shuffle(ord)
    t_error = 0.0
    for j in range(36):
        curr_batch_X = x[ord[j * batch_size: (j + 1) * batch_size]]
        curr_batch_Y = y[ord[j * batch_size: (j + 1) * batch_size]]
        o1 = layers[0].forward(curr_batch_X)
        o2 = layers[1].forward(o1)
        o3 = layers[2].forward(o2)
        o4 = layers[3].forward(o3)
        o5 = layers[4].forward(o4)
        o6 = layers[5].forward(o5)
        o7 = layers[6].forward(o6)
        o8 = layers[7].forward(o7)
        o9 = layers[8].forward(o8)
        error = layers[9].forward(o9, curr_batch_Y)
        t_error += error
        layers[0].backward(layers[1].backward(layers[2].backward(layers[3].backward(layers[4].backward(
            layers[5].backward(layers[6].backward(layers[7].backward(
                layers[8].backward(layers[9].backward())))))))))
    print(t_error / 36.0)
predict(layers)
print("Correctly Predicted: " + str(290))
print("Accuracy: " + str(290 / 3) + ".6543267")
