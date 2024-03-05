import csv

def ler_csv_remover_linhas_excedentes(nome_arquivo):
    conteudo = []
    with open(nome_arquivo, 'r', newline='') as arquivo_csv:
        leitor_csv = csv.reader(arquivo_csv)
        for linha in leitor_csv:
            if len(linha)!= 55:
                continue
            # else:
            #     conteudo.append(linha[:-1])


    with open(nome_arquivo, 'w', newline='') as arquivo_csv:
        escritor_csv = csv.writer(arquivo_csv)
        escritor_csv.writerows(conteudo)

nome_arquivo = 'DADOS_BRUTOS/CNN/data.csv'
ler_csv_remover_linhas_excedentes(nome_arquivo)
