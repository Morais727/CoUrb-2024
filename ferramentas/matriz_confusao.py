import glob
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

lista = []

try:
    modelos = ['DNN', 'CNN']
    niid_iid = ['NIID', 'IID']        
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']                        
    alpha_dirichlet = [0,0.1,0.5,2,5,10]
    noise_gaussiano = [0,0.1,0.5,0.8]
    round_inicio = [2, 4, 6, 8]
    per_cents_atacantes = [30,60,90,95]
            
    for i, j, k, l, m, n, o, p in product(niid_iid, ataques, data_set, modelos, per_cents_atacantes, alpha_dirichlet, noise_gaussiano, round_inicio):                    
        file_list = glob.glob(f'TESTES/{i}/LOG_ACERTOS/{j}_{k}_{l}_{m}_{n}_{o}_{p}.csv')  
        
        if file_list not in lista:      
            lista.append(file_list)
        
except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

for caminhos_arquivos in lista:
    rotulos = []
    
    for arquivo in caminhos_arquivos:
        try:
            arquivo = arquivo.replace('\\', '/')
             
            extensao = arquivo.split('.')
            caminho = '.'.join(extensao[:2]).split('/')
            base = caminho[3].split('_')
            rotulo = f'{base[-1]}'
            rotulos.append(rotulo)

            data = pd.read_csv(arquivo, header=None)

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
            plt.savefig(f'TESTES/{caminho[1]}/LOG_ACERTOS/GRAFICOS/{base[0]}_{base[1]}_{base[2]}_{base[3]}_{base[4]}_{base[5]}_{base[6]}_accuracy.png', dpi=300)
            plt.close('all')
        except Exception as e:
            print(f"Ocorreu um erro ao processar: {str(e)}")