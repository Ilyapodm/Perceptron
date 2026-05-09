import numpy as np
import math

class Perceptron:
    def __init__(self):
        self._w = np.random.randn(2, 1) * 0.01
        self._b = 0
        self._train_losses = []
        self._val_losses = []
        
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-z))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid((X @ self._w) + self._b)
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.clip(y_pred, 1e-8, 1 - 1e-8)  # исключаем y_true = 0 или 1 
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).sum()/len(y_true)
    
    # @staticmethod
    # def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    #     # все правильные предсказания на общее количество
    #     y_pred = y_pred >= 0.5
    #     accuracy = (y_pred == y_true)
    #     return 
        
        
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int,
            lr: int,
            batch_size: int):
        n_batches = math.ceil(len(X_train) / batch_size)  # остаток (если есть) тоже учитываем, np срезает до конца
        
        for epoch in range(epochs):
            # перемешиваем train
            
            loss_epoch = []
            for batch in range(n_batches):  # forward -> loss -> grad -> update
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size

                train_pred = self.forward(X_train[batch_start : batch_end])  # выход функции активации
                train_loss = self.compute_loss(y_train[batch_start : batch_end], train_pred)
                
                loss_epoch.append(train_loss)
                
                # dL/dz = train_pred - y - посчитал ручками аналитически
                grad_W = X_train[batch_start : batch_end].T @ (train_pred - y_train[batch_start : batch_end]) / batch_size
                grad_b = (train_pred - y_train[batch_start : batch_end]) / batch_size
                
                self._w -= lr * grad_W
                self._b -= lr * grad_b
                
            # конец эпохи
            self._train_losses.append(sum(loss_epoch) / n_batches)  # добавляем loss по train'у
            
            # val этап (не разбиваем на батчи, так как веса не обновляем)
            val_pred = self.forward(X_val)  
            val_loss = self.compute_loss(y_val, val_pred)
            
            self._train_losses.append(val_loss)  # добавляем loss val
            