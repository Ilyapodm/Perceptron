import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perceptron import Perceptron
from src.data_preprocessing import load_data
from helpers.helpers import print_metrics

EPOCHS = 100
LR = 0.1

X_train, y_train, X_val, y_val, X_test, y_test = load_data()  # загрузка данных

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))

batch_sizes = [1, 16, 64, 256]

# graphics utils
epoch_range = range(1, EPOCHS + 1)

for axe, batch_size in enumerate(batch_sizes):
    perceptron = Perceptron()
    
    perceptron.fit(X_train, y_train, X_val, y_val, EPOCHS, LR, batch_size)  # обучение
    
    # loss graphic
    axes[axe].plot(epoch_range, perceptron._train_losses, label="Train") 
    axes[axe].plot(epoch_range, perceptron._val_losses, label="Val")   
    axes[axe].set_xlabel('Epoch')
    axes[axe].set_ylabel('Loss')
    axes[axe].set_title(f'batch size = {batch_size}')
    axes[axe].legend()
    axes[axe].grid(True, alpha=0.3)
    
    axes[len(batch_sizes)].plot(epoch_range, perceptron._val_losses, label=f"batch size = {batch_size}")
    
    # metrics
    metrics_test = perceptron.evaluate(X_test, y_test)
    print_metrics(f"batch size = {batch_size}", metrics_test, time=perceptron._time)

axes[len(batch_sizes)].set_xlabel('Epoch')
axes[len(batch_sizes)].set_ylabel('Loss')
axes[len(batch_sizes)].set_title('Absolute compare')
axes[len(batch_sizes)].legend()
axes[len(batch_sizes)].grid(True, alpha=0.3)

plt.tight_layout() 
plt.show()