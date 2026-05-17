import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.perceptron import Perceptron
from src.data_preprocessing import load_data
from helpers.helpers import print_metrics

EPOCHS = 100
BATCH = 32

current_dir = Path(__file__)
project_dir = current_dir.parent.parent

plots_dir = project_dir / "report" / "experiment_plots"

plots_dir.mkdir(parents=True, exist_ok=True)

X_train, y_train, X_val, y_val, X_test, y_test = load_data()  # загрузка данных

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(15, 4))

learning_rates = [0.001, 0.01, 0.5, 1.0]

# graphics utils
epoch_range = range(1, EPOCHS + 1)
for axe, lr in enumerate(learning_rates):
    perceptron = Perceptron()
    
    perceptron.fit(X_train, y_train, X_val, y_val, EPOCHS, lr, BATCH)  # обучение
    
    # loss graphic
    axes[axe].plot(epoch_range, perceptron._train_losses, label="Train") 
    axes[axe].plot(epoch_range, perceptron._val_losses, label="Val")   
    axes[axe].set_xlabel('Epoch')
    axes[axe].set_ylabel('Loss')
    axes[axe].set_title(f'lr = {lr}')
    axes[axe].legend()
    axes[axe].grid(True, alpha=0.3)
    
    axes[len(learning_rates)].plot(epoch_range, perceptron._val_losses, label=f"lr={lr}")
    
    # metrics
    metrics_test = perceptron.evaluate(X_test, y_test)
    print_metrics(f"lr = {lr}", metrics_test, time=perceptron._time)

axes[len(learning_rates)].set_xlabel('Epoch')
axes[len(learning_rates)].set_ylabel('Loss')
axes[len(learning_rates)].set_title('Absolute compare')
axes[len(learning_rates)].legend()
axes[len(learning_rates)].grid(True, alpha=0.3)


plt.tight_layout() 

plt.savefig(plots_dir / "learning_rate_exp.jpg")
plt.show()