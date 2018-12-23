import numpy as np
from sklearn.utils import shuffle

classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


class smc_perceptron:
    def __init__(self, num_epochs, features_len, learning_rate):
        self.num_epochs = num_epochs
        self.w = np.zeros((26, features_len))
        self.learning_rate = learning_rate

    def train(self, training_set):
        for epoch in range(1, self.num_epochs):
            for xi, yi in training_set:
                y_hat = np.argmax(np.dot(self.w, xi))
                if yi != y_hat:
                    self.w[y_hat] += np.dot(self.learning_rate, np.dot(-1 ,xi))
                    self.w[yi] += np.dot(self.learning_rate, np.dot(1 ,xi))

class structured_perceptron:
    def __init__(self, num_epochs, features_len, learning_rate):
        self.num_epoch = num_epochs
        self.w = np.zeros((26,features_len))
        self.learning_rate = learning_rate
    
    def phi(self, x, y):
        zeros = np.zeros((26,128))
        zeros[y] = x
        return zeros

    def train(self, training_set):
        for epoch in range(1, self.num_epoch):
            for xi, yi in training_set:
                y_hat = self.find_argmax(xi)
                #print(y_hat)
                self.w += (self.phi(xi, yi) - self.phi(xi, y_hat))

    def find_argmax(self, x):
        temp_y = -1
        temp_max = [[-1 for i in range(26)] for j in range(26)]
        for i in range(26):
            temp_value = np.dot(self.w, np.transpose(self.phi(x, i)))
            if self.max_array(temp_value, temp_max):
                temp_max = temp_value
                temp_y = i
        return temp_y

    def find_argmax_good(self, x):
        array = []
        for i in range(26):
            array.append(np.dot(self.w, np.transpose(self.phi(x, i))))
        value = array.index(np.argmax(array))
        return value


    def max_array(self, temp_value,temp_max):
        comp = 0
        #print(len(temp_value))
        for i in range(len(temp_value)):
            for j in range(len(temp_value)):
                if temp_max[i][j] > temp_value[i][j] :
                    comp += 1
        if comp > len(temp_max)**2 / 2:
            return 0
        return 1

def predict(model, x):
    return np.argmax(np.dot(model.w, x))

def read_training_set():
    x_set = []
    y_set = []
    with open('letters.train.data') as f:
        for line in f:
            this_line = line.strip().split()
            label = this_line[1]
            data = [float(i) for i in this_line[6:]]
            x_set.append(np.array(data))
            y_set.append(classes.index(label))

    x_set, y_set = shuffle(x_set, y_set)
    training_set = list(zip(x_set,y_set))
    return training_set


def read_test_set():
    x_set = []
    y_set = []
    with open('letters.test.data') as f:
        for line in f:
            this_line = line.strip().split()
            label = this_line[1]
            data = [float(i) for i in this_line[6:]]
            x_set.append(np.array(data))
            y_set.append(classes.index(label))
    test_set = list(zip(x_set,y_set))
    return test_set

def predict_all(test_set, model):
    correct_answers = 0
    for xi, yi in test_set:
        pred = predict(model, xi)
        if pred == yi:
            correct_answers += 1
    print("The accuracy is: " + str((float(correct_answers) / len(test_set)) * 100) + "%")


print("Reading the training set...\n")

training_set = read_training_set()
temp_x, _ = training_set[0]
features_length = len(temp_x)
print("Training the model...\n")

#model = smc_perceptron(20, features_length, 0.01) # standart multiclass perceptron
model = structured_perceptron(20, features_length, 0.01) # structured multiclass perceptron

model.train(training_set)

print("Reading the test set...\n")

test_set = read_test_set()

print("Making predictions...\n")

predict_all(test_set, model)