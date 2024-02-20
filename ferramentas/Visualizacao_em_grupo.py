import os
import glob
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def calcular_media(lista):
    return sum(lista) / len(lista)

def condição_para_evitar_combinação(i, j, k, l, m, n, o, p):
    if k == 'MNIST' and l == 'CNN':
        return True
    if k == 'CIFAR10' and l == 'DNN':
        return True
    if i == 'IID' and n > 0:
        return True
    return False


tamanho_fonte = 25
lista = []
combinacoes_unicas = set()

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
       
        if condição_para_evitar_combinação(i, j, k, l, m, n, o, p):
            continue   
        
        file_list = glob.glob(f'TESTES/{i}/LOG_EVALUATE/{j}_{k}_{l}_{m}_{n}_{o}*.csv')  
        
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
            caminho = '.'.join(extensao[:3]).split('/')
            base = caminho[3].split('_')
            rotulo = f'{base[-1]}'
            rotulos.append(rotulo)
            
            plt.figure(figsize=(9, 5))
            for i, arquivo_atual in enumerate(caminhos_arquivos):
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

            plt.savefig(f'TESTES/{caminho[1]}/GRAFICOS/{base[0]}_{base[1]}_{base[2]}_{base[3]}_{base[4]}_{base[5]}_{base[6]}_accuracy.png', dpi=300)
            plt.close('all')

            plt.figure(figsize=(9, 5))
            for i, arquivo in enumerate(caminhos_arquivos):
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
                # title='# Round',
                title_fontsize=tamanho_fonte
            )

            plt.savefig(f'TESTES/{caminho[1]}/GRAFICOS/{base[0]}_{base[1]}_{base[2]}_{base[3]}_{base[4]}_{base[5]}_{base[6]}_loss.png', dpi=300)
            plt.close('all')
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")