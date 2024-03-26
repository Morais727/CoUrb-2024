import os
import glob
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def calcular_media(arquivos):
    return round(sum(arquivos) / len(arquivos), 2)

tamanho_fonte = 25
lista = []
combinacoes_unicas = []

try:
    modelos = ['DNN', 'CNN']
    niid_iid = ['IID','NIID']        
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']                        
    alpha_dirichlet = [0.0,0.5]
    noise_gaussiano = [0.0,0.1,]
    round_inicio = [4]
    per_cents_atacantes = [40]
            
   
         

    for i, j, k, l, m, n, o, p in product(niid_iid, ataques, data_set, modelos, per_cents_atacantes, alpha_dirichlet, noise_gaussiano, round_inicio):                    
        file_list = glob.glob(f'TESTES/{i}/LOG_EVALUATE/{j}_{k}_{l}_{m}_{n}_{o}*.csv') 
        combinacao = (i, j, k, l, m, n, o, p)  
            
        if i == 'IID' and n > 0:           
            continue

        if j != 'RUIDO_GAUSSIANO' and o > 0:        
            continue

        if (k == 'MNIST' and l == 'CNN') or (k == 'CIFAR10' and l == 'DNN'):            
            continue

        if combinacao not in combinacoes_unicas:                  
            combinacoes_unicas.append(combinacao)        
            lista.append(file_list)
        else:
                continue
except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

for arquivos in lista:
    rotulos = [] 
    for arquivo in arquivos:     
        try:
            arquivo = arquivo.replace('\\', '/')
            extensao = arquivo.split('.')
            caminho = '.'.join(extensao[:-1]).split('/') 
            nome_arquivo = caminho[3][:-1]             
            base = caminho[-1].split('_')
            rotulos.append(base[-1])
            
            plt.figure(figsize=(9, 5))
            for i, arquivo_atual in enumerate(arquivos):
                data = pd.read_csv(arquivo_atual, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'accuracy': calcular_media,
                }).reset_index()
                
                
                plt.plot(media_round['server_round'], media_round['accuracy'], label=f'{rotulos[i]}', linewidth=3)
                
                
            xticks = np.arange(0,21,2)
            plt.xticks(xticks, fontsize=tamanho_fonte)
            plt.xticks(fontsize=tamanho_fonte)
            plt.yticks(fontsize=tamanho_fonte)

            plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(
                loc='best',
                fontsize=tamanho_fonte,
                ncol=1,
                title='# Round',
                title_fontsize=tamanho_fonte
            )
            arquivo_accuracy = f'TESTES/{caminho[1]}/GRAFICOS/{nome_arquivo}_accuracy.png'
            os.makedirs(os.path.dirname(arquivo_accuracy), exist_ok=True)
            plt.savefig(arquivo_accuracy, dpi=100)
            plt.close()

            plt.figure(figsize=(9, 5))
            for i, arquivo in enumerate(arquivos):
                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'loss': calcular_media,
                }).reset_index()
                
                plt.plot(media_round['server_round'], media_round['loss'], label=f'{rotulos[i]}', linewidth=3)

            xticks = np.arange(0,21,2)
            plt.xticks(xticks, fontsize=tamanho_fonte)
            plt.xticks(fontsize=tamanho_fonte)
            plt.yticks(fontsize=tamanho_fonte)
            
            plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(
                loc='best',
                fontsize=tamanho_fonte,
                ncol=1,
                title='# Round',
                title_fontsize=tamanho_fonte
            )
            arquivo_loss = f'TESTES/{caminho[1]}/GRAFICOS/{nome_arquivo}_loss.png'
            os.makedirs(os.path.dirname(arquivo_loss), exist_ok=True)
            plt.savefig(arquivo_loss, dpi=100)
            plt.close()
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")
            a=e
       