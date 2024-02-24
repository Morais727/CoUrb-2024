import os
import csv
import glob
import shutil


def substituir_valor_contagem(texto):
    return texto.replace('Valor,Contagem', '')


lista = glob.glob(f'TESTES/NID/LABELS/*.csv')


for arquivo in lista:
    try:
        arquivo_temporario = arquivo

        with open(arquivo, 'r', newline='') as arquivo_csv:
            leitor_csv = csv.reader(arquivo_csv)
            linhas = [substituir_valor_contagem(linha) for linha in leitor_csv]

        with open(arquivo_temporario, 'w', newline='') as novo_arquivo_csv:
            escritor_csv = csv.writer(novo_arquivo_csv)
            for linha in linhas:
                escritor_csv.writerow(linha)

        shutil.move(arquivo_temporario, arquivo)
        
    except Exception as e:
        print(f"Ocorreu um erro ao processar: {str(e)}")
