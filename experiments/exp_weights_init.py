import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perceptron import Perceptron
from src.data_preprocessing import load_data
from helpers.helpers import print_metrics

EPOCHS = 100
BATCH = 32
LR = 0.1
SEED = 42
X_train, y_train, X_val, y_val, X_test, y_test = load_data()  # загрузка данных

fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(15, 4))

np.random.seed(SEED)

weights = {
    'zeros': np.zeros((2, 1)),
    'small': None,  # дефолт
    'large': np.random.randn(2, 1) * 10
}

# graphics utils
epoch_range = range(1, EPOCHS + 1)

for axe, dict_info in enumerate(weights.items()):
    name, weight = dict_info
    
    perceptron = Perceptron(weights=weight)
    
    perceptron.fit(X_train, y_train, X_val, y_val, EPOCHS, LR, BATCH)  # обучение
    
    # loss graphic
    axes[axe].plot(epoch_range, perceptron._train_losses, label="Train") 
    axes[axe].plot(epoch_range, perceptron._val_losses, label="Val")   
    axes[axe].set_xlabel('Epoch')
    axes[axe].set_ylabel('Loss')
    axes[axe].set_title(f'weights = {name}')
    axes[axe].legend()
    axes[axe].grid(True, alpha=0.3)
    
    axes[len(weights)].plot(epoch_range, perceptron._val_losses, label=f"weights = {name}")
    
    # metrics
    metrics_test = perceptron.evaluate(X_test, y_test)
    print_metrics(f"weights = {name}", metrics_test, time=perceptron._time)

axes[len(weights)].set_xlabel('Epoch')
axes[len(weights)].set_ylabel('Loss')
axes[len(weights)].set_title('Absolute compare')
axes[len(weights)].legend()
axes[len(weights)].grid(True, alpha=0.3)

plt.tight_layout() 
plt.show()