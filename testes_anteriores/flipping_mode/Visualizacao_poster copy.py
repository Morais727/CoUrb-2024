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
lista =  [   
                ['flipping_mode/random_Fraction_fit_iid_mnist_atak_2.csv',
                'flipping_mode/random_Fraction_fit_iid_mnist_atak_4.csv',
                'flipping_mode/random_Fraction_fit_iid_mnist_atak_6.csv',
                'dflipping_mode/random_Fraction_fit_iid_mnist_atak_8.csv',
                ],[
                'flipping_mode/random_Fraction_fit_iid_mnist_dnn_2.csv',
                'flipping_mode/random_Fraction_fit_iid_mnist_dnn_4.csv',
                'flipping_mode/random_Fraction_fit_iid_mnist_dnn_6.csv',
                'flipping_mode/frandom_Fraction_fit_iid_mnist_dnn_8.csv',
                ]
            ]

rotulos = []

# Criar subplots para Accuracy e Loss
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))
for arquivos in lista:
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
            # plt.legend(
            #     loc='best',
            #     fontsize=tamanho_fonte,
            #     ncol=1,
            #     title='# Round',
            #     title_fontsize=tamanho_fonte
            # )
            arquivo_accuracy = f'flipping_mode/GRAFICOS/{arquivo}_accuracy.png'
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
            # plt.legend(
            #     loc='best',
            #     fontsize=tamanho_fonte,
            #     ncol=1,
            #     title='# Round',
            #     title_fontsize=tamanho_fonte
            # )
            arquivo_loss = f'flipping_mode/GRAFICOS/{arquivo}_loss.png'
            os.makedirs(os.path.dirname(arquivo_loss), exist_ok=True)
            plt.savefig(arquivo_loss, dpi=100)
            plt.close()
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")
            a=e
       