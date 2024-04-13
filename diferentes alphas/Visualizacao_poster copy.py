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
                'TESTES/TESTES/DATA/normal_Fraction_Fit_iid_mnist_dnn_0.0.csv',
                'TESTES/TESTES/DATA/random_Fraction_Fit_iid_mnist_dnn_0.4.csv',
                'TESTES/TESTES/DATA/random_Fraction_Fit_iid_mnist_dnn_0.6.csv',
                'TESTES/TESTES/DATA/random_Fraction_Fit_iid_mnist_dnn_0.8.csv',
                'TESTES/TESTES/DATA/random_Fraction_Fit_iid_mnist_dnn_1.0.csv',
            ]

rotulos = []

# Criar subplots para Accuracy e Loss
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

for i, metrica in enumerate(['accuracy', 'loss']):
    # Cria um único gráfico para Accuracy ou Loss com várias linhas
    for j, arquivo in enumerate(arquivos):
        csv =  arquivo.split('.')
        base = csv[0].split('/')
        rotulo = base[3].split('_')
        rotulos.append(f'{rotulo[6]}.{csv[1]}')

        data = pd.read_csv(arquivo, header=None)
        data.columns = ['server_round', 'cid', 'accuracy', 'loss']
        media_round = data.groupby('server_round').agg({
            metrica: calcular_media,
        }).reset_index()

        axes[i].plot(media_round['server_round'], media_round[metrica], label=f'{rotulos[j]}', linewidth=3)

    xticks = np.arange(0,51,10)
    axes[i].set_xticks(xticks)
    axes[i].tick_params(axis='both', which='major', labelsize=tamanho_fonte)
    axes[i].grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
    if metrica == 'loss' :
        axes[i].legend(
            loc='best',
            fontsize=tamanho_fonte,
            ncol=1,
            title='Alpha',
            title_fontsize=tamanho_fonte
        )

# Salva a figura
plt.tight_layout()
plt.savefig('comparacao_normal_iid_mnist_dnn_combined.png', dpi=200)
plt.show()
