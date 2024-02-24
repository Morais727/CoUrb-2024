import glob
import random
import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt

tamanho_fonte = 20
lista = []
try:
    modelos = ['DNN', 'CNN']
    niid_iid = ['NIID']        
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']                        
    alpha_dirichlet = [0,0.1,0.5,2,5,10]
    noise_gaussiano = [0,0.1,0.5,0.8]
    round_inicio = [2, 4, 6, 8]
    per_cents_atacantes = [30,60,90,95]
    
    lista = set()
    combinacoes_unicas = set()        
    for i, j, k, l, m, n, o, p in product(niid_iid, ataques, data_set, modelos, per_cents_atacantes, alpha_dirichlet, noise_gaussiano, round_inicio):                    
        file_list = glob.glob(f'TESTES/{i}/LABELS/*.csv') 
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
    
except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

for file in lista:
    try: 
        file = file.replace('\\', '/') 

        extensao = file.split('.')
        caminho = '.'.join(extensao[:-1]).split('/')
        base = caminho[3].split('_')

        column_names = ['user', 'labels']

        df = pd.read_csv(file, names=column_names)
        total_por_usuario = df.groupby('user')['labels'].count()
        quantidade_labels = df.groupby('user')['labels'].nunique()
        
        df_plot = df.groupby(['user', 'labels']).size().reset_index().pivot(columns='labels', index='user', values=0)

        percentual = df_plot.div(total_por_usuario, axis=0) 
        
        # Correção na linha abaixo
        df_plot_percent = percentual * quantidade_labels.values[:, np.newaxis]
        xticks = np.arange(0,21, 2)
        plt.xticks(xticks, fontsize = tamanho_fonte)
        plt.yticks(fontsize = tamanho_fonte)

        fig, ax = plt.subplots(figsize=(20, 8))
        df_plot_percent.plot(kind='bar', stacked=True, ax=ax, legend=None)

        plt.grid(color='0.9', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
        
        fig = plt.gcf()
        fig.set_size_inches(12, 5)
        
        fig.savefig(f'TESTES/{caminho[1]}/LABELS/GRAFICOS/{base[0]}_{base[1]}_{base[2]}_{base[3]}_{base[4]}_{base[5]}_{base[6]}_{base[7]}.png', dpi = 100,bbox_inches = 'tight', pad_inches = 0.05)
        plt.close('all')
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo {file}: {str(e)}")
