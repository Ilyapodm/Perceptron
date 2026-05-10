import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocessing import load_data
from src.perceptron import Perceptron
from helpers.helpers import print_metrics

LR = 0.1
EPOCHS = 100
BATCH = 32



X_train, y_train, X_val, y_val, X_test, y_test = load_data()  # загрузка данных

perceptron = Perceptron()

perceptron.fit(X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH)  # обучение

metrics_train = perceptron.evaluate(X_train, y_train)
metrics_test = perceptron.evaluate(X_test, y_test)

epoch_range = range(1, EPOCHS + 1)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))

# axes - массив объектов осей
axes[0].plot(epoch_range, perceptron._train_losses, label="Train") 
axes[0].plot(epoch_range, perceptron._val_losses, label="Val")   
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('BCE Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(metrics_test.get("ROC")[0], metrics_test.get("ROC")[1])
axes[1].set_xlabel('FPR')
axes[1].set_ylabel('TPR')
axes[1].set_title('ROC')
axes[1].grid(True, alpha=0.3)

x1 = np.linspace(X_test[:, 0].min(), X_test[:, 0].max(), 100)

axes[2].plot(x1, - (perceptron._w[0, 0] * x1 + perceptron._b) / perceptron._w[1, 0])  #x2​=−(w1​* x1​+b)/w2​.
axes[2].scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='bwr')
axes[2].set_xlabel('x1')
axes[2].set_ylabel('x2')
axes[2].set_title('График точек')
axes[2].grid(True, alpha=0.3)   


print_metrics("Train", metrics_train, time=perceptron._time)
print_metrics("Test",  metrics_test)

plt.tight_layout() 
plt.show()

