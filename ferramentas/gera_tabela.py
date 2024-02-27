import os
import glob
import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def calcular_media(lista):
    return round(sum(lista) / len(lista), 2)

def criar_tabela(media_geral_acuracia, titulo, caminho_destino):
    fig, ax = plt.subplots()
    ax.axis('off')

    table_data = [['Modelo', 'Média de Acurácia']]
    for modelo, media_acuracia in media_geral_acuracia.items():
        table_data.append([modelo, media_acuracia])

    # Verificar se a lista table_data está vazia antes de tentar criar a tabela
    if len(table_data) > 1:
        table = ax.table(cellText=table_data, loc='center', cellLoc='center', colLabels=table_data.pop(0))
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.2)

        plt.title(titulo)
        
        # Verificar se o diretório de destino existe, se não, criá-lo
        if not os.path.exists(caminho_destino):
            os.makedirs(caminho_destino)
        
        plt.savefig(f'{caminho_destino}/{titulo}_accuracy.png', dpi=300)
    else:
        print("Nenhum dado encontrado para criar a tabela.")

tamanho_fonte = 25
lista_iid_dnn = []
lista_iid_cnn = []
lista_niid_dnn = []
lista_niid_cnn = []

try:
    niid_iid = ['IID', 'NIID']
    ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
    data_set = ['MNIST', 'CIFAR10']
    modelos = ['DNN', 'CNN']

    for i, j, k, l in product(niid_iid, ataques, data_set, modelos):
        file_list = glob.glob(f'TESTES/{i}/LOG_EVALUATE/{j}_{k}_{l}_*.csv')
        if i == 'IID' and  l == 'DNN':        
            lista_iid_dnn.append((j, l, file_list))
        elif i == 'IID' and l == 'CNN':
            lista_iid_cnn.append((j, l, file_list))        
        elif i == 'NIID' and l == 'DNN':
            lista_niid_dnn.append((j, l, file_list))
        elif i == 'NIID' and l == 'CNN':
            lista_niid_cnn.append((j, l, file_list))

except Exception as e:
    print(f"Ocorreu um erro ao processar: {str(e)}")

# Dicionário para armazenar médias de acurácia de cada modelo
media_acuracia_por_modelo_iid_dnn = {}
media_acuracia_por_modelo_iid_cnn = {}
media_acuracia_por_modelo_niid_dnn = {}
media_acuracia_por_modelo_niid_cnn = {}

# Processar os dados e calcular média de acurácia para cada modelo de ataque

# Processar os dados e calcular média de acurácia para cada modelo de ataque
def processar_arquivos(lista, media_acuracia_por_modelo):
    for ataque, modelo, arquivos in lista:
        chave = f'{ataque}_{modelo}'  # Usar uma chave única que leve em consideração o ataque e o modelo
        if chave not in media_acuracia_por_modelo:
            media_acuracia_por_modelo[chave] = []
        for arquivo in arquivos:
            try:
                arquivo = arquivo.replace('\\', '/')
                extensao = arquivo.split('.')
                caminho = '.'.join(extensao[:-1]).split('/')
                base = caminho[-1].split('_')
                rotulo = f'{base[-1]}'

                data = pd.read_csv(arquivo, header=None)
                data.columns = ['server_round', 'cid', 'accuracy', 'loss']
                media_round = calcular_media(data['accuracy'])
                
                media_acuracia_por_modelo[chave].append(media_round)

            except Exception as e:
                print(f"Ocorreu um erro ao processar o arquivo {arquivo}: {str(e)}")


# Processar os arquivos IID para DNN
processar_arquivos(lista_iid_dnn, media_acuracia_por_modelo_iid_dnn)

# Processar os arquivos NIID para DNN
processar_arquivos(lista_niid_dnn, media_acuracia_por_modelo_niid_dnn)

# Processar os arquivos IID para CNN
processar_arquivos(lista_iid_cnn, media_acuracia_por_modelo_iid_cnn)

# Processar os arquivos NIID para CNN
processar_arquivos(lista_niid_cnn, media_acuracia_por_modelo_niid_cnn)

# Calcular a média geral de acurácia para cada modelo e tipo
media_geral_acuracia_iid_dnn = {ataque: calcular_media(media_acuracias) for ataque, media_acuracias in media_acuracia_por_modelo_iid_dnn.items()}
media_geral_acuracia_niid_dnn = {ataque: calcular_media(media_acuracias) for ataque, media_acuracias in media_acuracia_por_modelo_niid_dnn.items()}
media_geral_acuracia_iid_cnn = {ataque: calcular_media(media_acuracias) for ataque, media_acuracias in media_acuracia_por_modelo_iid_cnn.items()}
media_geral_acuracia_niid_cnn = {ataque: calcular_media(media_acuracias) for ataque, media_acuracias in media_acuracia_por_modelo_niid_cnn.items()}

# Criar tabelas individuais para cada combinação de tipo (IID/NIID) e modelo (DNN/CNN)
criar_tabela(media_geral_acuracia_iid_dnn, 'IID - DNN', 'TESTES/IID/GRAFICOS')
criar_tabela(media_geral_acuracia_niid_dnn, 'NIID - DNN', 'TESTES/NIID/GRAFICOS')
criar_tabela(media_geral_acuracia_iid_cnn, 'IID - CNN', 'TESTES/IID/GRAFICOS')
criar_tabela(media_geral_acuracia_niid_cnn, 'NIID - CNN', 'TESTES/NIID/GRAFICOS')
