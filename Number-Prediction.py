import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, callbacks, regularizers
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and configured for memory growth.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available.")

n_epochs = 20
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 1
random_seed = 1
tf.random.set_seed(random_seed)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255

"""
#Train
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(train_images)
model = models.Sequential([
    layers.Conv2D(10, kernel_size=5, activation='relu', input_shape=(28, 28, 1),
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(20, kernel_size=5, activation='relu',
                  kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
os.makedirs('results', exist_ok=True)
model_checkpoint = callbacks.ModelCheckpoint('results/best_mnist_model.keras', monitor='val_loss', save_best_only=True)
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


train_losses = []
train_counter = []
test_losses = []
test_counter = []


class CustomCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % log_interval == 0:
            print(f'Train Epoch: {epoch + 1} Loss: {logs["loss"]:.6f} Train Accuracy: {logs["accuracy"]:.2%}')
            train_losses.append(logs['loss'])
            train_counter.append((epoch + 1) * len(train_images))
            self.model.save('results/model.keras')
            
            test_loss, test_accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
            test_losses.append(test_loss)
            test_counter.append((epoch + 1) * len(train_images))
            print(f'Test Accuracy: {test_accuracy:.2%}')

# Train the model with data augmentation
history = model.fit(datagen.flow(train_images, train_labels, batch_size=batch_size_train),
                    epochs=n_epochs, validation_data=(test_images, test_labels),
                    callbacks=[CustomCallback(), model_checkpoint, early_stopping])
"""

model = tf.keras.models.load_model('results/best_mnist_model.keras')

train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nTrain Accuracy: {train_acc:.2%}')
print(f'Train Loss: {train_loss:.4f}')
print(f'Test Accuracy: {test_acc:.2%}')
print(f'Test Loss: {test_loss:.4f}')


examples = test_images[:6]
predictions = model.predict(examples)

fig = plt.figure()
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(examples[i].reshape(28, 28), cmap='gray', interpolation='none')
    plt.title(f'Prediction: {np.argmax(predictions[i])}')
    plt.xticks([])
    plt.yticks([])
plt.show()
