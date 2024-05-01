import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

tamanho_fonte = 25

# Função para calcular média de uma lista
def calcular_media(lista):
    return sum(lista) / len(lista)

# Ler os dados do arquivo CSV
lista = [
    [
        'LOG_EVALUATE/ALTERNA_INICIO_CIFAR10_CNN_2.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_CIFAR10_CNN_4.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_CIFAR10_CNN_6.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_CIFAR10_CNN_8.csv',
    ],
    [
        'LOG_EVALUATE/ALTERNA_INICIO_MNIST_DNN_2.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_MNIST_DNN_2.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_MNIST_DNN_2.csv',
        'LOG_EVALUATE/ALTERNA_INICIO_MNIST_DNN_2.csv',
    ],
    [
        'LOG_EVALUATE/ATACANTES_CIFAR10_CNN_2.csv',
        'LOG_EVALUATE/ATACANTES_CIFAR10_CNN_4.csv',
        'LOG_EVALUATE/ATACANTES_CIFAR10_CNN_6.csv',
        'LOG_EVALUATE/ATACANTES_CIFAR10_CNN_8.csv',
    ],
    [
        'LOG_EVALUATE/INVERTE_TREINANDO_CIFAR10_CNN_6.csv',
        'LOG_EVALUATE/INVERTE_TREINANDO_CIFAR10_CNN_4.csv',
        'LOG_EVALUATE/INVERTE_TREINANDO_CIFAR10_CNN_6.csv',
        'LOG_EVALUATE/INVERTE_TREINANDO_CIFAR10_CNN_8.csv',
    ]
]

for arquivos in lista:
    rotulos = []
    for arquivo in arquivos:
        try:
            base = arquivo.split('.')
            ponto = base[0].split('_')

            rotulos.append(ponto[-1])

            # Cria um único gráfico para Accuracy com várias linhas
            plt.figure(figsize=(9, 5))

            # Listas para armazenar as médias e os desvios padrão
            medias = []
            desvios_padrao = []

            for i, arquivo in enumerate(arquivos):
                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = data.groupby('server_round').agg({
                    'accuracy': calcular_media,
                }).reset_index()

                medias.append(media_round['accuracy'].mean())  # Calcula a média das accuracies
                desvios_padrao.append(media_round['accuracy'].std())  # Calcula o desvio padrão das accuracies

                # Plota a linha de cada rodada com cor mais fraca
                plt.plot(media_round['server_round'], media_round['accuracy'], alpha=0.1, color='gray')

            # Plota a linha da média com cor mais forte
            plt.plot(media_round['server_round'], medias, label=f'Média {rotulos[i]}', linewidth=3)

            # Calcula as linhas de erro
            lower_bound = np.array(medias) - np.array(desvios_padrao)
            upper_bound = np.array(medias) + np.array(desvios_padrao)

            # Plota as linhas de erro com cor mais forte
            plt.fill_between(media_round['server_round'], lower_bound, upper_bound, alpha=0.2, color='blue')

            xticks = np.arange(0, 101, 10)
            plt.xticks(xticks, fontsize=tamanho_fonte)
            plt.xticks(fontsize=tamanho_fonte)
            plt.yticks(fontsize=tamanho_fonte)

            plt.grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
            plt.legend(loc='best', fontsize=tamanho_fonte, ncol=1, title='# Round', title_fontsize=tamanho_fonte)

            # Salva o gráfico de Accuracy
            nome1 = f'comparacao_{base[0]}_accuracy.png'
            os.makedirs(os.path.dirname(nome1), exist_ok=True)
            plt.savefig(nome1, dpi=100)
            plt.close('all')

        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")
