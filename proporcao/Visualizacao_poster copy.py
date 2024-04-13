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
arquivos = [
    'TESTES/IID/DATA_REGULAR/normal_iid_mnist_dnn_20%.csv',
    'TESTES/IID/DATA_REGULAR/normal_iid_mnist_dnn_40%.csv',
    'TESTES/IID/DATA_REGULAR/normal_iid_mnist_dnn_60%.csv',
    'TESTES/IID/DATA_REGULAR/normal_iid_mnist_dnn_80%.csv',
    'TESTES/IID/DATA_REGULAR/random_iid_mnist_dnn_20%.csv',
    'TESTES/IID/DATA_REGULAR/random_iid_mnist_dnn_40%.csv',
    'TESTES/IID/DATA_REGULAR/random_iid_mnist_dnn_60%.csv',
    'TESTES/IID/DATA_REGULAR/random_iid_mnist_dnn_80%.csv',
]

estilo_linha = ['--', '--', '--', '--', '-', '-', '-', '-']
cor_linha = ['blue', 'red', 'green', 'purple', 'blue', 'red', 'green', 'purple']

rotulos = ['random_20%', 'random_40%', 'random_60%', 'random_80%', 'normal_20%', 'normal_40%', 'normal_60%', 'normal_80%']

# Criar subplots para Accuracy e Loss
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

for i, metrica in enumerate(['accuracy', 'loss']):
    # Cria um único gráfico para Accuracy ou Loss com várias linhas
    for j, arquivo in enumerate(arquivos):
        data = pd.read_csv(arquivo, header=None)
        data.columns = ['server_round', 'cid', 'accuracy', 'loss']
        media_round = data.groupby('server_round').agg({
            metrica: calcular_media,
        }).reset_index()

        axes[i].plot(media_round['server_round'], media_round[metrica], label=f'{rotulos[j]}', linestyle=estilo_linha[j], color=cor_linha[j], linewidth=3)

    xticks = np.arange(1, 11)
    axes[i].set_xticks(xticks)
    axes[i].tick_params(axis='both', which='major', labelsize=tamanho_fonte)
    axes[i].grid(color='k', linestyle='--', linewidth=0.5, axis='both', alpha=0.1)
    if metrica == 'loss' :
        axes[i].legend(
            loc='best',
            fontsize=tamanho_fonte,
            ncol=1,
            title='% Ataque',
            title_fontsize=tamanho_fonte
        )

# Salva a figura
plt.tight_layout()
plt.savefig('comparacao_normal_iid_mnist_dnn_combined.png', dpi=200)
plt.show()
