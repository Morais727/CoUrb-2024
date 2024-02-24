import os
import glob

# Função para substituir 'Valor,Contagem' por uma string vazia
def substituir_valor_contagem(texto):
    return texto.replace('Valor,Contagem', '')

# Diretório dos arquivos
diretorio = 'TESTES/NIID/LABELS'

# Listar arquivos CSV no diretório
arquivos = glob.glob(os.path.join(diretorio, '*.csv'))

# Iterar sobre cada arquivo
for arquivo in arquivos:
    try:
        with open(arquivo, 'r', newline='') as f:
            # Ler todo o conteúdo do arquivo
            conteudo = f.read()
        
        # Substituir a expressão no conteúdo
        conteudo_modificado = substituir_valor_contagem(conteudo)
        
        # Escrever o conteúdo modificado de volta para o arquivo
        with open(arquivo, 'w', newline='') as f:
            f.write(conteudo_modificado)

        print(f"Expressão substituída em {arquivo}.")
    except Exception as e:
        print(f"Ocorreu um erro ao processar {arquivo}: {str(e)}")
