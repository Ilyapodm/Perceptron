import numpy as np
import math
import time

class Perceptron:
    def __init__(self):
        self._w = np.random.randn(2, 1) * 0.01
        self._b = 0
        self._train_losses = []
        self._val_losses = []
        self._val_accuracy = []
        
    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-z))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid((X @ self._w) + self._b)
    
    @staticmethod
    def compute_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)  # исключаем y_true = 0 или 1 
        return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        
    
    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int,
            lr: float,
            batch_size: int):
        start = time.perf_counter()
        n_batches = math.ceil(len(X_train) / batch_size)  # остаток (если есть) тоже учитываем, np корректно срезает до конца
        
        for epoch in range(epochs):
            # перемешиваем train
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            loss_epoch = []
            for batch in range(n_batches):  # forward -> loss -> grad -> update
                batch_start = batch * batch_size
                batch_end = (batch + 1) * batch_size

                train_pred = self.forward(X_train[batch_start : batch_end])  # выход функции активации
                train_loss = self.compute_loss(y_train[batch_start : batch_end], train_pred)
                
                loss_epoch.append(train_loss)
                
                # dL/dz = train_pred - y - посчитал ручками аналитически
                grad_W = X_train[batch_start : batch_end].T @ (train_pred - y_train[batch_start : batch_end]) / batch_size
                grad_b = (train_pred - y_train[batch_start : batch_end]).sum() / batch_size
                
                self._w -= lr * grad_W
                self._b -= lr * grad_b
                
            # конец эпохи
            self._train_losses.append(sum(loss_epoch) / n_batches)  # добавляем train loss
            
            # val этап (не разбиваем на батчи и не перемешиваем, т.к. веса не обновляем)
            val_pred = self.forward(X_val)  
            val_loss = self.compute_loss(y_val, val_pred)
            
            val_bin_pred = self.binary_prediction(val_pred)
            checked_val = self.check_answer(y_val, val_bin_pred)
            val_accuracy = self.compute_accuracy(checked_val)  # добавляем val accuracy
            self._val_accuracy.append(val_accuracy)
            
            self._val_losses.append(val_loss)  # добавляем val loss
        
        end = time.perf_counter()
        self._time = end - start
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X)
        return self.binary_prediction(y_pred)
        
    # Метрики
    @staticmethod
    def binary_prediction(y_pred: np.ndarray, threshold=0.5) -> np.ndarray:
        return y_pred >= threshold    
    
    @staticmethod
    def check_answer(y_true: np.ndarray, y_bin_pred: np.ndarray) -> np.ndarray:
        """ возвращает массив true/false - правильное или неправильное предсказание"""
        return y_bin_pred == y_true
    
    @staticmethod
    def compute_TP(y_true: np.ndarray, y_bin_pred: np.ndarray):
        return (y_bin_pred & y_true).sum()
        
    @staticmethod
    def compute_TN(y_true: np.ndarray, y_bin_pred: np.ndarray):
        return len(y_true) - (y_bin_pred | y_true).sum()
    
    @staticmethod
    def compute_FP(y_true: np.ndarray, y_bin_pred: np.ndarray):
        return (y_bin_pred & (~y_true)).sum()
    
    @staticmethod
    def compute_FN(y_true: np.ndarray, y_bin_pred: np.ndarray):
        return ((~y_bin_pred) & y_true).sum()
    
    @staticmethod
    def compute_accuracy(checked_y: np.ndarray) -> float:
        """ Все правильные предсказания на общее количество: (TP + TN) / TP + TN + FP + FN"""
        return checked_y.sum() / len(checked_y)
    
    @staticmethod
    def compute_precision(TP: int, FP: int) -> float:
        """
        сколько из положительных предсказанных - реально положительных?
        TP / (TP + FP)
        """
        return TP / (TP + FP)
    
    @staticmethod
    def compute_recall(TP, FN) -> float:
        """
        сколько из действительно положительных модель угадала положительных
        TP / (TP + FN)
        """
        return TP / (TP + FN)
    
    @staticmethod
    def compute_f1(precision: float, recall: float) -> float:
        if (precision + recall) == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
        
    @staticmethod
    def compute_ROC_AUC(y_true: np.ndarray, y_pred: np.ndarray):
        """
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        """
        fpr, tpr = [], []
        # измеряем только там, где будут изменения, обратный порядок для корректного вычисления площади AUC
        thresholds = np.sort(np.unique(y_pred))[::-1]   
        
        for threshold in thresholds:
            y_bin_pred = Perceptron.binary_prediction(y_pred, threshold)  # предсказываем класс по каждому порогу
            TP = Perceptron.compute_TP(y_true, y_bin_pred)
            TN = Perceptron.compute_TN(y_true, y_bin_pred)
            FP = Perceptron.compute_FP(y_true, y_bin_pred)
            FN = Perceptron.compute_FN(y_true, y_bin_pred)
            TPR = TP / (TP + FN)
            FPR = FP / (FP + TN)
            
            tpr.append(TPR)
            fpr.append(FPR)
    
        fpr = [0.0] + fpr + [1.0]
        tpr = [0.0] + tpr + [1.0]

        auc = np.trapezoid(tpr, fpr)
        
        return (fpr, tpr, auc)
        
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        # возвращает словарь с метриками
        metrics = {}
        y = y.astype(bool)  # приводим к bool для корректных битовых операций
        y_pred = self.forward(X)  # вероятности
        y_bin_pred = self.binary_prediction(y_pred)  # массив bool (наше предсказание классов 0/1)
        checked_y = Perceptron.check_answer(y, y_bin_pred)  # правильные/неправильные ответы, удобно для accuracy
        
        TP = self.compute_TP(y, y_bin_pred)
        FP = self.compute_FP(y, y_bin_pred)
        FN = self.compute_FN(y, y_bin_pred)
        
        metrics["accuracy"] = self.compute_accuracy(checked_y)
        metrics["precision"] = self.compute_precision(TP, FP)
        metrics["recall"] = self.compute_recall(TP, FN)
        metrics["f1"] = self.compute_f1(metrics["precision"], metrics["recall"])
        fpr, tpr, auc = self.compute_ROC_AUC(y, y_pred)
        metrics["ROC"] = (fpr, tpr)
        metrics["AUC"] = auc
        metrics["time"] = self._time
        return metrics
    