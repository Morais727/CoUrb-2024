import os
import glob
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def calcular_media(lista):
    return sum(lista) / len(lista)

tamanho_fonte = 25
lista = []

try:
    niid_iid = ['IID', 'NIID']
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']
    modelos = ['DNN', 'CNN']

    for i, j, k, l in product(niid_iid, ataques, data_set, modelos):
        file_list = glob.glob(f'TESTES/{i}/LOG_EVALUATE/{j}_{k}_{l}*.csv')
        lista.append(file_list)

except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

for caminhos_arquivos in lista:
    rotulos = []
    for arquivo in caminhos_arquivos:
        try:
            arquivo = arquivo.replace('\\', '/')
            
            extensao = arquivo.split('.')
            caminho = extensao[0].split('/')
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

            xticks = np.arange(0,101,10)
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

            plt.savefig(f'TESTES/{caminho[1]}/GRAFICOS/{caminho[4]}_accuracy.jpg')
            plt.close('all')

            plt.figure(figsize=(9, 5))
            for i, arquivo in enumerate(caminhos_arquivos):
                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'loss': calcular_media,
                }).reset_index()

                plt.plot(media_round['server_round'], media_round['loss'], label=f'{rotulos[i]}', linewidth=3)

            xticks = np.arange(0,101,10)
            plt.xticks(xticks, fontsize=tamanho_fonte)
            plt.xticks(fontsize=tamanho_fonte)
            plt.yticks(fontsize=tamanho_fonte)
            
            plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(
                loc='best',
                fontsize=tamanho_fonte,
                ncol=1,
                title='Round',
                title_fontsize=tamanho_fonte
            )

            plt.savefig(f'TESTES/{caminho[1]}/GRAFICOS/{caminho[4]}_loss.jpg')
            plt.close('all')
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")
