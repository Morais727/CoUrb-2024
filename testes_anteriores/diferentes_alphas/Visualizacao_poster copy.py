import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import os
import glob

# Função para calcular média de uma lista
def calcular_media(lista):
    return sum(lista) / len(lista)

tamanho_fonte = 25

# Ler os dados do arquivo CSV
arquivos =  [   
                'diferentes_alphas/DATA/normal_Fraction_Fit_iid_mnist_dnn_0.0.csv',
                'diferentes_alphas/DATA/random_Fraction_Fit_iid_mnist_dnn_0.4.csv',
                'diferentes_alphas/DATA/random_Fraction_Fit_iid_mnist_dnn_0.6.csv',
                'diferentes_alphas/DATA/random_Fraction_Fit_iid_mnist_dnn_0.8.csv',
                'diferentes_alphas/DATA/random_Fraction_Fit_iid_mnist_dnn_1.0.csv',
            ]

rotulos = []

# Criar subplots para Accuracy e Loss
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

for arquivo in arquivos:     
    try:
        
        
        rotulos= ['0.0','0.1','0.4','0.6','0.8','1.0' ]
        
        # plt.figure(figsize=(9, 5))
        plt.figure(figsize=(10, 7))
        for i, arquivo_atual in enumerate(arquivos):
            data = pd.read_csv(arquivo_atual, header=None)
            data.columns = ['server_round', 'cid', 'accuracy', 'loss']
            media_round = data.groupby('server_round').agg({
                'accuracy': calcular_media,
            }).reset_index()
            
            
            plt.plot(media_round['server_round'], media_round['accuracy'], label=f'{rotulos[i]}', linewidth=3)
            
            
        xticks = np.arange(0,51,10)
        plt.xticks(xticks, fontsize=tamanho_fonte)
        plt.xticks(fontsize=tamanho_fonte)
        plt.yticks(fontsize=tamanho_fonte)
        plt.ylabel('Accuracy', fontsize=tamanho_fonte)
        plt.xlabel('Round', fontsize=tamanho_fonte)
        # plt.xlabel('# Round', fontsize=tamanho_fonte, labelpad=20)
        # plt.ylabel('Loss', fontsize=tamanho_fonte, labelpad=10)
        plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
        plt.legend(
            loc='lower right',
            fontsize=tamanho_fonte,
            ncol=1,
            title='Alfa',
            title_fontsize=tamanho_fonte
        )
        arquivo_accuracy = f'diferentes_alphas/GRAFICOS/{arquivo}_accuracy.png'
        os.makedirs(os.path.dirname(arquivo_accuracy), exist_ok=True)
        plt.savefig(arquivo_accuracy, dpi=100)
        plt.close()

        plt.figure(figsize=(10, 7))
        for i, arquivo in enumerate(arquivos):
            data = pd.read_csv(arquivo, header=None)
            data.columns = ['server_round', 'cid', 'accuracy', 'loss']
            media_round = data.groupby('server_round').agg({
                'loss': calcular_media,
            }).reset_index()
            
            plt.plot(media_round['server_round'], media_round['loss'], label=f'{rotulos[i]}', linewidth=3)

        xticks = np.arange(0,51,10)
        plt.xticks(xticks, fontsize=tamanho_fonte)
        plt.xticks(fontsize=tamanho_fonte)
        plt.yticks(fontsize=tamanho_fonte)
        plt.ylabel('Loss', fontsize=tamanho_fonte)
        plt.xlabel('Round', fontsize=tamanho_fonte)
        # plt.xlabel('# Round', fontsize=tamanho_fonte, labelpad=20)
        # plt.ylabel('Loss', fontsize=tamanho_fonte, labelpad=10)
        plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
        plt.legend(
            loc='lower left',
            fontsize=tamanho_fonte,
            ncol=1,
            title='Alfa',
            title_fontsize=tamanho_fonte
        )
        arquivo_loss = f'diferentes_alphas/GRAFICOS/{arquivo}_loss.png'
        os.makedirs(os.path.dirname(arquivo_loss), exist_ok=True)
        plt.savefig(arquivo_loss, dpi=100)
        plt.close()
    except Exception as e:
        print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")
        a=e
       