import pandas as pd

# [IA] registro N - Adaptação dos tratamentos existentes da Frente 1 para funções.
# Não foram adicionadas novas decisões de limpeza.
def limpar_pedidos(df):
    dados = df.copy()
    dados.columns = [
        'id_pedido', 'id_cliente', 'status_pedido', 'marca_temporal_compra_pedido',
        'pedido_aprovado_as', 'data_entrega_pedido_transporadora',
        'data_entrega_pedido_cliente', 'data_estimada_entrega_pedido'
    ]
    dados.columns = dados.columns.str.strip().str.lower().str.replace(' ', '_')
    datas = ['marca_temporal_compra_pedido', 'pedido_aprovado_as',
            'data_entrega_pedido_transporadora', 'data_entrega_pedido_cliente',
            'data_estimada_entrega_pedido']
    for coluna in datas:
        dados[coluna] = pd.to_datetime(dados[coluna], errors='coerce')
    compra = dados['marca_temporal_compra_pedido']
    transportadora = dados['data_entrega_pedido_transporadora']
    entrega = dados['data_entrega_pedido_cliente']
    datas_erradas = (transportadora < compra) | (transportadora > entrega)
    dados.loc[datas_erradas, 'data_entrega_pedido_transporadora'] = pd.NaT
    return dados


def limpar_itens(df):
    dados = df.copy()
    dados.columns = ['id_pedido', 'id_item_pedido', 'id_produto', 'id_vendedor',
                    'data_limite_envio', 'preco', 'valor_frete']
    dados['data_limite_envio'] = pd.to_datetime(dados['data_limite_envio'], errors='coerce')
    return dados


def limpar_produtos(df):
    dados = df.copy()
    dados.columns = ['id_produto', 'nome_categoria_produto', 'comprimento_nome_produto',
                    'comprimento_descricao_produto', 'qtd_fotos_produto',
                    'peso_produto_g', 'comprimento_produto_cm', 'altura_produto_cm',
                    'largura_produto_cm']
    dados = dados.dropna(subset=['id_produto'])
    for coluna in ['comprimento_nome_produto', 'comprimento_descricao_produto', 'qtd_fotos_produto']:
        dados[coluna] = dados[coluna].astype('Int64')
    return dados


def limpar_avaliacoes(df):
    dados = df.copy()
    dados.columns = ['id_avaliacao', 'id_pedido', 'nota_avaliacao',
                    'titulo_comentario_avaliacao', 'mensagem_comentario_avaliacao',
                    'data_criacao_avaliacao', 'marca_temporal_resposta_avaliacao']
    dados['titulo_comentario_avaliacao'] = dados['titulo_comentario_avaliacao'].fillna('Vazio')
    dados['mensagem_comentario_avaliacao'] = dados['mensagem_comentario_avaliacao'].fillna('Vazio')
    dados['data_criacao_avaliacao'] = pd.to_datetime(dados['data_criacao_avaliacao'], errors='coerce')
    dados['marca_temporal_resposta_avaliacao'] = pd.to_datetime(dados['marca_temporal_resposta_avaliacao'], errors='coerce')
    return dados


def limpar_clientes(df):
    dados = df.copy()
    dados.columns = ['id_cliente', 'id_unica_cliente', 'prefixo_cep_cliente',
                    'cidade_cliente', 'estado_cliente']
    return dados
