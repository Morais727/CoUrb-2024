import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

data = pd.read_csv('TESTES/IID/LOG_ACERTOS/ALTERNA_INICIO_MNIST_DNN_60_0_0_2.csv', header=None)

situacao = data[2].astype(int)
prev = data[3].apply(lambda x: int(x.strip('[]'))) 

cm = confusion_matrix(situacao, prev)

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
classes = ['Negativo', 'Positivo']  
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

fmt = 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.ylabel('Classe Real')
plt.xlabel('Classe Prevista')
plt.tight_layout()
plt.show()
