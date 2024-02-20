import os
import glob
file_list = set()
modos = ['IID','NIID']
for i in modos:
    list = glob.glob(f'TESTES/{i}/LOG_EVALUATE/*.csv')
    file_list.update(list)
for arquivo in file_list:
        try:
            novo_nome = arquivo.replace('[', '').replace(']', '')
            os.rename(arquivo, novo_nome)
        except Exception as e:
            print(f"Ocorreu um erro ao processar: {str(e)}")