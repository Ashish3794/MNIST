import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns 

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train_full, y_train_full), (x_test, y_test) = mnist.load_data()

print(f"Size of x_train: {x_train_full.shape}")
print(f"Size of x_test: {x_test.shape}")

#Noramlized the value between 0 and 1
x_valid, x_train = x_train_full[:5000]/255., x_train_full[5000:]/255.
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
x_test = x_test/255

layers = [
    tf.keras.layers.Flatten(input_shape = [28,28], name = "input_layer"),
    tf.keras.layers.Dense(300, activation = "relu", name = "hidden_layer1"),
    tf.keras.layers.Dense(100, activation = "relu", name = "hidden_layer2"),
    tf.keras.layers.Dense(10, activation = "softmax", name = "output_layer")
]

model_clf = tf.keras.models.Sequential(layers)

weights, biases = model_clf.layers[1].get_weights()

LOSS_FUNCTION = "sparse_categorical_crossentropy"
OPTIMIZER = "SGD"
METRICS = ["accuracy"]
EPOCHS = 30
VALIDATION = (x_valid, y_valid)

model_clf.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)
history = model_clf.fit(x_train, y_train, epochs = EPOCHS, validation_data = VALIDATION)

pd.DataFrame(history.history).plot(figsize = (10,7))
plt.grid(True)
plt.show()

model_clf.evaluate(x_test, y_test)

x_new = x_test[:3]
y_prob = model_clf.predict(x_new)
y_prob.round(3)

y_pred = np.argmax(y_prob, axis = 1)

for img_array, pred, actual in zip(x_new, y_pred, y_test[:3]):
    plt.imshow(img_array, cmap = "binary")
    plt.title(f"Predicted: {pred}, Actual: {actual}")
    plt.axis("off")
    plt.show()
    print("-----"*20)

model_clf.save("model.h5")
