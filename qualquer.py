import numpy as np
import tensorflow as tf
# for i in range(3):
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
unique_classes, counts = np.unique(y_train, return_counts=True)

# print("Classes existentes:", unique_classes)
print("Quantidades relativas:", counts)

num_classes = len(unique_classes)
alpha = [0.1] * num_classes

sample = np.random.dirichlet(alpha)

target = (counts * sample).astype(int)
X_train, y_train = X_train[target], y_train[target]

for class_label, num_images in zip(unique_classes, target):
    print(f"Classe {class_label}: {num_images} imagens")