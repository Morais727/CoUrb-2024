import os
import glob
file_list = set()
modos = ['IID','NIID']
locais = ['LOG_EVALUATE','LOG_ACERTOS','LOG_ACERTOS','LABELS' ]
for i in modos:
    for j in locais:
        list = glob.glob(f'TESTES/{i}/{j}/*.csv')
        file_list.update(list)
for arquivo in file_list:
        try:
            novo_nome = arquivo.replace('[', '').replace(']', '')
            os.rename(arquivo, novo_nome)
        except Exception as e:
            print(f"Ocorreu um erro ao processar: {str(e)}")