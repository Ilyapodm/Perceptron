import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_data
from perceptron import Perceptron

LR = 0.1
EPOCHS = 100
BATCH = 32

X_train, y_train, X_val, y_val, X_test, y_test = load_data()  # загрузка данных

perceptron = Perceptron()

perceptron.fit(X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH)  # обучение

epoch_range = range(1, EPOCHS + 1)

plt.plot(epoch_range, perceptron._train_losses, label="Train")
plt.plot(epoch_range, perceptron._val_losses, label="Val")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('BCE Loss')
plt.legend()
plt.grid(True)
plt.show()