import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Load Fashion MNIST Dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and reshape data
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test  = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Model builder function
def build_model(filter_size=3, reg_rate=0.0, optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, (filter_size, filter_size), activation='relu', input_shape=(28, 28, 1),
                     kernel_regularizer=l2(reg_rate)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(reg_rate)))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    if optimizer == 'adam':
        opt = Adam()
    elif optimizer == 'sgd':
        opt = SGD()
    elif optimizer == 'rmsprop':
        opt = RMSprop()
    
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Different configurations to test
configs = [
    {'name': 'Filter 3x3', 'filter_size': 3, 'reg_rate': 0.0, 'batch_size': 64, 'optimizer': 'adam'},
    {'name': 'Filter 5x5', 'filter_size': 5, 'reg_rate': 0.0, 'batch_size': 64, 'optimizer': 'adam'},
    {'name': 'L2 Regularization', 'filter_size': 3, 'reg_rate': 0.01, 'batch_size': 64, 'optimizer': 'adam'},
    {'name': 'Small Batch Size', 'filter_size': 3, 'reg_rate': 0.0, 'batch_size': 32, 'optimizer': 'adam'},
    {'name': 'Optimizer SGD', 'filter_size': 3, 'reg_rate': 0.0, 'batch_size': 64, 'optimizer': 'sgd'},
]

# Train and evaluate models
results = []
for config in configs:
    print(f"\nTraining: {config['name']}")
    model = build_model(config['filter_size'], config['reg_rate'], config['optimizer'])
    history = model.fit(x_train, y_train, epochs=5, batch_size=config['batch_size'], verbose=0,
                        validation_data=(x_test, y_test))
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{config['name']} - Test Accuracy: {test_acc:.4f}")
    results.append((config['name'], test_acc))

# Plot results
names = [r[0] for r in results]
scores = [r[1] for r in results]

plt.figure(figsize=(10,5))
plt.barh(names, scores, color='skyblue')
plt.xlabel("Accuracy")
plt.title("Effect of Different Hyperparameters on CNN Performance")
plt.xlim(0, 1)
plt.grid(True)
plt.show()
