import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Função para calcular média de uma lista
def calcular_media(lista):
    return sum(lista) / len(lista)

tamanho_fonte = 20
# Ler os dados do arquivo CSV   random_Fraction_fit_niid_cifar_cnn_0.csv',
lista = [   
            [   
                'random_Fraction_fit_iid_mnist_NOVO_2.csv',
                'random_Fraction_fit_iid_mnist_NOVO_4.csv',
                'random_Fraction_fit_iid_mnist_NOVO_6.csv',
                'random_Fraction_fit_iid_mnist_NOVO_8.csv',
                'random_Fraction_fit_iid_mnist_NOVO_10.csv',
            ],
            [   
                
                'random_Fraction_fit_niid_mnist_dnn_2.csv',
                'random_Fraction_fit_niid_mnist_dnn_4.csv',
                'random_Fraction_fit_niid_mnist_dnn_6.csv',
                'random_Fraction_fit_niid_mnist_dnn_8.csv',
                'random_Fraction_fit_niid_mnist_dnn_10.csv',
            ]
        ]

for arquivos in lista:
    rotulos = []
    for arquivo in arquivos:
        try:      
            csv =  arquivo.split('.')
            base = csv[0].split('_')
            rotulos.append(base[6])
           
            
            # Cria um único gráfico para Accuracy com várias linhas
            plt.figure(figsize=(9, 5))
            for i, arquivo in enumerate(arquivos):                    
                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'accuracy': calcular_media,
                }).reset_index()

                plt.plot(media_round['server_round'], media_round['accuracy'], label=f'{rotulos[i]}', linewidth = 3)

            xticks = np.arange(0,21, 2)
            plt.xticks(xticks, fontsize = tamanho_fonte)
            plt.yticks(fontsize = tamanho_fonte)

            plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(
                loc='best',
                fontsize= tamanho_fonte,
                ncol = 1,
                title = 'Round',
                title_fontsize = tamanho_fonte
            )

            # Salva o gráfico de Accuracy
            fig = plt.gcf()
            fig.set_size_inches(12, 5)
            fig.savefig(f'comparacao_{base[5]}_accuracy.png', dpi = 500,bbox_inches = 'tight', pad_inches = 0.05)
            plt.close('all')

            # Cria um único gráfico para Loss com várias linhas
            plt.figure(figsize=(9, 5))
            for i, arquivo in enumerate(arquivos):                   
                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'loss': calcular_media,
                }).reset_index()

                plt.plot(media_round['server_round'], media_round['loss'], label=f'{rotulos[i]}', linewidth = 3)

            plt.grid(color='0.9', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(
                loc='best',
                fontsize= tamanho_fonte,
                ncol = 1,
                title = 'Round',
                title_fontsize = tamanho_fonte
            )

            # Salva o gráfico de Loss
            plt.savefig(f'comparacao_{base[5]}_loss.png', dpi = 500,bbox_inches = 'tight', pad_inches = 0.05)
            plt.close('all')
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")