import glob
import shlex
import subprocess 
import concurrent.futures
from itertools import product

limpa_arquivos_csv= []
padroes =   [
                'TESTES/IID/LABELS/*.csv', 
                'TESTES/IID/LOG_EVALUATE/*.csv', 
                'TESTES/IID/LOG_ACERTOS/*.csv',
                'TESTES/NIID/LABELS/*.csv', 
                'TESTES/NIID/LOG_EVALUATE/*.csv',  
                'TESTES/NIID/LOG_ACERTOS/*.csv',
                
            ]

for i in padroes:
    limpa_arquivos_csv.extend(glob.glob(i))

try:
    for arquivo in limpa_arquivos_csv:
        with open(arquivo, 'w') as file:
            pass
except subprocess.CalledProcessError as e:
    print(f'Erro: {e}')

arquivos_teste= [
                    'simulacao_principal.py'
                ]


def executar_arquivo(arquivo):

    try:
        niid_iid = ['IID', 'NIID']
        ataques = ['ALTERNA_INICIO', 'ATACANTES', 'EMBARALHA', 'INVERTE_TREINANDO', 'INVERTE_SEM_TREINAR', 'INVERTE_CONVEGENCIA', 'ZEROS', 'RUIDO_GAUSSIANO', 'NORMAL']
        data_set = ['MNIST', 'CIFAR10']
        modelos = ['DNN', 'CNN']
        variaveis = [2, 4, 6, 8]

        for i, j, k, l, m in product(niid_iid, ataques, data_set, modelos, variaveis):
            print(f'Executando {arquivo}')
            
            if k == 'MNIST' and l == 'CNN':
                print(f'Combinação inválida: Dataset {k} com modelo {l}. Pulando execução.')
                continue
            elif k == 'CIFAR10' and l == 'DNN':
                print(f'Combinação inválida: Dataset {k} com modelo {l}. Pulando execução.')
                continue
            
            comando = f'python3 {arquivo} --iid_niid {i} --modo_ataque {j} --dataset {k} --modelo_definido {l} --variavel {m}'
            print(f'\n\n################################################################################################')
            print(f'\n\n{comando}\n\n')
            print(f'################################################################################################\n\n')
            subprocess.run(shlex.split(comando), check=True)

            print(f'Executou com sucesso: {arquivo}')

    except subprocess.CalledProcessError as e:
        print(f'Erro: {arquivo}')

max_threads = 1

with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    resultados = list(executor.map(executar_arquivo, arquivos_teste))    
