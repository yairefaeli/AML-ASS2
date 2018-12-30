import numpy as np
from sklearn.utils import shuffle

classes = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
num_of_letters = 26



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

    def predict(self, x):
        return np.argmax(np.dot(self.w, x))

class structured_perceptron:
    def __init__(self, num_epochs, features_len, learning_rate):
        self.num_epoch = num_epochs
        self.features_len = features_len
        self.w = np.zeros(num_of_letters * features_len)
        self.learning_rate = learning_rate
    
    def phi(self, x, y):
        zeros = np.zeros(num_of_letters * self.features_len)
        z = int(y * self.features_len)
        zeros[z:z + self.features_len] = x
        return zeros

    def train(self, training_set):
        for epoch in range(1, self.num_epoch):
            for xi, yi in training_set:
                y_hat = self.find_argmax(xi)
                self.w = self.w + (self.phi(xi, yi) - self.phi(xi, y_hat))

    def find_argmax(self, x):
        return np.argmax([np.dot(self.w, self.phi(x, y)) for y in range(num_of_letters)])

    def predict(self, x):
        return self.find_argmax(x)

class dps_perceptron:
    def __init__(self, num_epochs, features_len):
        self.num_epochs = num_epochs
        self.features_len = features_len
        size = (features_len * num_of_letters) + ((num_of_letters + 1) * (num_of_letters + 1))
        self.w = np.zeros(size)

    def train(self, training_set):
         for epoch in range(1, self.num_epoch):
            for xi, yi in training_set:
                y_hat = self.find_argmax(xi)
                self.w += (self.phi_for_word(xi, yi) - self.phi_for_word(xi, y_hat))

    def phi_for_word(self, x, y):
        size = (self.features_len * num_of_letters) + ((num_of_letters + 1) * (num_of_letters + 1))
        zeros = np.zeros(size)
        prev_y = -1
        for i in range(len(y)):
            zeros += self.phi2(x[i], prev_y, int(y[i]))
            prev_char = int(y[i])
        return zeros

    def phi(self, x, y):
        size = (self.features_len * num_of_letters) + ((num_of_letters + 1) * (num_of_letters + 1))
        zeros = np.zeros(size)
        z = int(y * self.features_len)
        zeros[z:z + self.features_len] = x
        return zeros        

    def phi2(self, x, prev_y, y):
        size = (self.features_len * num_of_letters) + ((num_of_letters + 1) * (num_of_letters + 1))
        zeros = np.zeros(size)
        z = int(y * self.features_len)
        zeros[z:z + self.features_len] = x
        z = int(self.features_len * num_of_letters + prev_y * (num_of_letters + 1) + y)
        zeros[z] = 1
        return zeros

    def find_argmax(self, x):
        word_len = len(x)
        d_s = np.zeros((word_len, num_of_letters))
        d_pi = np.zeros((word_len, num_of_letters))
        prev_y = -1 # for $
        for i in range(num_of_letters):
            self.d_s[0][i] = np.dot(self.w, self.phi2(x[0], prev_y, i))
            self.d_pi[0][i] = 0
        
        for i in range(1, word_len):
            for j in range(num_of_letters):
                letter = j
                max_value = -1
                max_index = -1
                for k in range(num_of_letters):
                    temp_value = np.dot(self.w, self.phi2(x[i], k, letter)) + self.d_s[i-1][k]
                    if temp_value > max_value:
                        max_value = temp_value
                        max_index = k
                self.d_s[i][j] = max_value
                self.d_pi[i][j] = max_index
        
        y_hat = np.zeros(word_len)
        max_value = -1
        for i in range(num_of_letters):
            if max_value < d_s[word_len - 1][i]:
                y_hat[word_len - 1] = i
                max_value = d_s[word_len - 1][i]

        for i in range(word_len - 2, -1, -1):
            y_hat[i] = d_pi[i + 1][int(y_hat[i + 1])]

        return y_hat

    def predict(self, x):
        return self.find_argmax(x)

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

# HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def read_training_set_by_word():
    None

def read_test_set_by_word():
    None


def predict_all(test_set, model):
    correct_answers = 0
    for xi, yi in test_set:
        pred = model.predict(xi)
        if pred == yi:
            correct_answers += 1
    print("The accuracy is: " + str((float(correct_answers) / len(test_set)) * 100) + "%")

print("Reading the training set...\n")

#training_set = read_training_set()
training_set = read_test_set_by_word()
temp_x, _ = training_set[0]
features_length = len(temp_x)
print("Training the model...\n")

#model = smc_perceptron(20, features_length, 0.01) # standart multiclass perceptron
#model = structured_perceptron(10, features_length, 0.01) # structured multiclass perceptron
model = dps_perceptron(10, features_length) # dynamic programming perceptron.

model.train(training_set)

print("Reading the test set...\n")

#test_set = read_test_set()
test_set = read_test_set_by_word()

print("Making predictions...\n")

predict_all(test_set, model)