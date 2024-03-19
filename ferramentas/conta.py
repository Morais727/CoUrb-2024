import os
import glob
import pandas as pd
from sklearn.utils import resample

erros = []

pastas = ['data dnn', 'data cnn']

for pasta in pastas:
    lista_de_dfs = []
    try:
        arquivos = glob.glob(f'DADOS_BRUTOS/{pasta}/*.csv', recursive=True)
        print(f"Arquivos encontrados na pasta {pasta}: {arquivos}")  # Adicionando um print para verificar os arquivos encontrados

        for arquivo in arquivos:
            try:
                data = pd.read_csv(arquivo, header=None)

                if data.empty:
                    erros.append(f'ERRO: O arquivo {arquivo} está vazio.')
                    continue

                lista_de_dfs.append(data)

            except Exception as e:
                erros.append(f'ERRO {arquivo}: {str(e)}')
        print(erros)  # Adicionando um print para verificar os erros encontrados durante a leitura dos arquivos

        if lista_de_dfs:
            df_mestre = pd.concat(lista_de_dfs, ignore_index=True)                        
            cont = df_mestre.iloc[:, -1].value_counts()
            indice = df_mestre.shape[1] - 1
            atak = df_mestre[df_mestre[indice] == 0]
            n_atak = df_mestre[df_mestre[indice] == 1]
            menor = min(len(atak), len(n_atak))
            

            if menor > 0:
                atak_undersampled = resample(atak, n_samples=menor)
                n_atak_undersampled = resample(n_atak, n_samples=menor)

                conjunto_balanceado = pd.concat([atak_undersampled, n_atak_undersampled], ignore_index=True)

                conjunto_balanceado = conjunto_balanceado.sample(frac=1)

                conjunto_balanceado.to_csv(f'DADOS_BRUTOS/Balanceado_{pasta}.csv', header=None, index=False)
                print(f"Dados balanceados salvos com sucesso para a pasta {pasta}!")  # Adicionando um print para verificar se os dados balanceados foram salvos

            print(cont)
        else:
            print(f"Nenhum arquivo encontrado para concatenação na pasta {pasta}.")

    except Exception as e:
        erros.append(f'ERRO ao processar arquivos na pasta {pasta}: {str(e)}')
        print(erros)
