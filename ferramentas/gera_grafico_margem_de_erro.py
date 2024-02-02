import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns 
import pandas as pd
import numpy as np
import glob

sns.set_theme(style='darkgrid')

lista = []
try:
    niid_iid = ['IID', 'NIID']
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']
    modelos = ['DNN', 'CNN']

    for i, j, k, l in product(niid_iid, ataques, data_set, modelos):
        file_list = glob.glob(f'TESTES/rodou_dez_vezes/{i}/LOG_EVALUATE/{j}_{k}_{l}*.csv')
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

            for i, arquivo_atual in enumerate(caminhos_arquivos):
                data = pd.read_csv(arquivo_atual, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']

                sns.lineplot(x='server_round', y='accuracy',
                data=data)
            plt.savefig(f'TESTES/{caminho[1]}/GRAFICOS/{caminho[4]}_accuracy.jpg')
            plt.close('all')
        except Exception as e:
            print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")




plt.show()