import os
import glob
import shlex
import subprocess 
import concurrent.futures
from itertools import product

limpa_arquivos_csv = []
caminhos = [ 'NIID', 'IID']
for caminho in caminhos:
    padroes = [

            f'TESTES/{caminho}/LABELS/*.csv', 
            f'TESTES/{caminho}/LOG_EVALUATE/*.csv', 
            f'TESTES/{caminho}/LOG_ACERTOS/*.csv',
            f'TESTES/{caminho}/LOG_ACERTOS/GRAFICOS/*.png',
            f'TESTES/{caminho}/GRAFICOS/*.png',
            f'TESTES/{caminho}/GRADIENTES/*/*.csv',
            ]

    for i in padroes:
        limpa_arquivos_csv.extend(glob.glob(i))

arquivos_teste = ['simulacao_principal.py']

def executar_arquivo(arquivo):
    try:
        num_round = [20]
        total_clients = [20]
        modelos = ['CNN','DNN']
        niid_iid = ['NIID', 'IID']      
        ataques = ['ZEROS', 'INVERTE_CONVEGENCIA','INVERTE_TREINANDO']
        data_set = ['MNIST', 'CIFAR10']                        
        alpha_dirichlet = [0.0,0.5]
        noise_gaussiano = [0.1,0.0]
        round_inicio = [2,4,6,8]
        per_cents_atacantes = [40]
        

        combinacoes_unicas = set() 

        
        try:
            for arquivo in limpa_arquivos_csv:
                os.remove(arquivo)

        except OSError as e:
            print(f'Erro ao limpar arquivo: {e}')

        for i, j, k, l, m, n, o, p, q, r in product(niid_iid, ataques, data_set, modelos, round_inicio, per_cents_atacantes, noise_gaussiano, alpha_dirichlet, num_round, total_clients):
            combinacao = (i, j, k, l, m, n, o, p, q, r) 
            
            if i == 'NIID' and p == 0:                             
                continue
            elif i == 'IID' and p > 0:               
                continue
            elif j != 'RUIDO_GAUSSIANO' and o > 0:   
                continue
            elif j == 'RUIDO_GAUSSIANO' and o != 0:
                continue            
            elif (k == 'MNIST' and l == 'CNN') or (k == 'CIFAR10' and l == 'DNN'):
                continue
            else:
                if combinacao not in combinacoes_unicas:           
                    combinacoes_unicas.add(combinacao)
         

            print(f'Executando {arquivo}')                
            comando = f'python3 {arquivo} --iid_niid {i} --modo_ataque {j} --dataset {k} --modelo_definido {l} --round_inicio {m} --per_cents_atacantes {n} --noise_gaussiano {o} --alpha_dirichlet {p} --num_rounds {q}  --total_clients {r}'
                    
            print(f'\n\n################################################################################################')
            print(f'\n\n{comando}\n\n')
            print(f'################################################################################################\n\n')
            subprocess.run(shlex.split(comando), check=True)

            print(f'Executou com sucesso: {arquivo}')

    except subprocess.CalledProcessError as e:
        print(f'Erro ao executar o arquivo: {e}')
    except Exception as e:
        print(f'Erro inesperado: {e}')


def processar_dados(arquivos):
    for arquivo in arquivos:
        try:
            subprocess.run(['python3', arquivo], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar {arquivo}: {e}")


max_threads = 1

with concurrent.futures.ThreadPoolExecutor(max_threads) as executor:
    resultados = list(executor.map(executar_arquivo, arquivos_teste))

def processar_graficos(arquivos):
    for arquivo in arquivos:
        try:
            subprocess.run(['python3', arquivo], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar {arquivo}: {e}")

gera_graficos = ['ferramentas/retira_caracter_nome.py','ferramentas/matriz_confusao.py', 'ferramentas/Visualizacao_em_grupo.py','ferramentas/gera_grafico_niid.py','ferramentas/gera_tabela.py','ferramentas/mede_similaridade.py']

processar_dados(gera_graficos)