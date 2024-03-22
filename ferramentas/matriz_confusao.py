import os
import glob
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

try:
    modelos = ['DNN', 'CNN']
    niid_iid = ['IID','NIID']        
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']                        
    alpha_dirichlet = [0.0]
    noise_gaussiano = [0.0, 0.1]
    round_inicio = [4,8]
    per_cents_atacantes = [40]

    lista = set()
    combinacoes_unicas = set()        

    for i, j, k, l, m, n, o, p in product(niid_iid, ataques, data_set, modelos, per_cents_atacantes, alpha_dirichlet, noise_gaussiano, round_inicio):                    
        file_list = glob.glob(f'TESTES/{i}/LOG_ACERTOS/*.csv') 
        combinacao = (i, j, k, l, m, n, o, p)  
           
        if i == 'IID' and n > 0: 
                   
            continue
             
        if j != 'RUIDO_GAUSSIANO' and o > 0:        
            continue

        if (k == 'MNIST' and l == 'CNN') or (k == 'CIFAR10' and l == 'DNN'):            
            continue
        
        if combinacao not in combinacoes_unicas:                            
            combinacoes_unicas.add(combinacao)        
            lista.update(file_list)
        else:
                continue
             
except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

for arquivo in lista:
    try:
        plt.figure()  # Criar uma nova figura para cada iteração do loop
        
        arquivo = arquivo.replace('\\', '/')
        extensao = arquivo.split('.')
        caminho = '.'.join(extensao[:-1]).split('/')        
        base = caminho[-1].split('_')
       
        data = pd.read_csv(arquivo, header=None)        

        situacao = data[2].astype(int)
        prev = data[3].apply(lambda x: int(x.strip('[]')) if isinstance(x, str) else x)


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
        arquivo_ = f'TESTES/{caminho[1]}/LOG_ACERTOS/GRAFICOS/{base[0]}_{base[1]}_{base[2]}_{base[3]}_{base[4]}_{base[5]}_{base[6]}_{base[7]}_MATRIZ_CONFUSAO.png'
        os.makedirs(os.path.dirname(arquivo_), exist_ok=True)
        plt.savefig(arquivo_, dpi=300)
        plt.close()
    except Exception as e:
        print(f"Ocorreu um erro ao processar: {str(e)}")
