import csv

# Função para ler o arquivo CSV e remover campos extras
def processar_csv(entrada, saida):
    with open(entrada, 'r', newline='') as arquivo_entrada, \
         open(saida, 'w', newline='') as arquivo_saida:
        leitor_csv = csv.reader(arquivo_entrada)
        escritor_csv = csv.writer(arquivo_saida)

        for linha in leitor_csv:
            # Verifica se a linha tem mais de 37 campos
            if len(linha) != 37:
                # Remove os campos extras (do 38 em diante)
                continue
            elif len(linha) >37 and linha[37] != int:
                continue
            else:                
                escritor_csv.writerow(linha)

# Chamada da função
processar_csv('DADOS_BRUTOS/NIID/DNN.csv', 'DADOS_BRUTOS/NIID/DNN_corrigido.csv')
