import pandas as pd
import numpy as np

def montar_base(pedidos, itens, produtos, avaliacoes, clientes):
    aval = avaliacoes.copy()
    aval['data_criacao_avaliacao'] = pd.to_datetime(aval['data_criacao_avaliacao'])
    aval['marca_temporal_resposta_avaliacao'] = pd.to_datetime(aval['marca_temporal_resposta_avaliacao'])

    # Conta as avaliações e mantém a mais recente de cada pedido.
    quantidade = aval.groupby('id_pedido').size().rename('qtd_avaliacoes')
    aval = (aval.sort_values(['data_criacao_avaliacao', 'marca_temporal_resposta_avaliacao', 'id_avaliacao'])
                .drop_duplicates('id_pedido', keep='last'))

    itens = itens.merge(produtos[['id_produto', 'nome_categoria_produto']], on='id_produto', how='left')

    # [IA] registro 1 - juntar os itens sem repetir o pedido
    # Agrupa os itens antes do merge para não repetir os valores de preço e frete.
    totais = itens.groupby('id_pedido').agg(
        qtd_itens=('id_item_pedido', 'size'),
        qtd_vendedores=('id_vendedor', 'nunique'),
        valor_produtos=('preco', 'sum'),
        valor_frete=('valor_frete', 'sum')
    ).reset_index()

    # Se tiver vários produtos, considera a categoria do item mais caro.
    categoria = (itens.sort_values('preco', ascending=False)
                    .drop_duplicates('id_pedido')[['id_pedido', 'nome_categoria_produto']]
                    .rename(columns={'nome_categoria_produto': 'categoria_produto'}))
    totais = totais.merge(categoria, on='id_pedido', how='left')

    # Junta avaliações, pedidos, valores e informações dos clientes.
    base = aval.merge(pedidos, on='id_pedido', how='left')
    base = base.merge(totais, on='id_pedido', how='left')
    base = base.merge(clientes[['id_cliente', 'estado_cliente']], on='id_cliente', how='left')
    base = base.merge(quantidade, on='id_pedido', how='left')
    return base

def criar_atributos(base):
    df = base.copy()
    datas = ['marca_temporal_compra_pedido', 'data_entrega_pedido_transporadora',
            'data_entrega_pedido_cliente', 'data_estimada_entrega_pedido']
    for coluna in datas:
        df[coluna] = pd.to_datetime(df[coluna], errors='coerce')

    compra = df['marca_temporal_compra_pedido']
    transportadora = df['data_entrega_pedido_transporadora']
    entrega = df['data_entrega_pedido_cliente']
    previsao = df['data_estimada_entrega_pedido']

    # [IA] registro 2 - tempo de entrega e atraso
    # Calcula quanto tempo o pedido levou e se chegou depois do prazo.
    df['tempo_entrega_dias'] = (entrega - compra).dt.total_seconds().div(86400).round(2)
    df['tempo_ate_transportadora_dias'] = (transportadora - compra).dt.total_seconds().div(86400).round(2)
    df['tempo_transporte_dias'] = (entrega - transportadora).dt.total_seconds().div(86400).round(2)
    df['desvio_prazo_dias'] = (entrega.dt.normalize() - previsao.dt.normalize()).dt.days
    df['dias_atraso'] = df['desvio_prazo_dias'].clip(lower=0)
    df['atrasou_entrega'] = (df['desvio_prazo_dias'] > 0).where(df['desvio_prazo_dias'].notna()).astype('Int64')

    # Soma produtos e frete e calcula a proporção do frete.
    df['valor_total_pedido'] = df['valor_produtos'] + df['valor_frete']
    df['proporcao_frete'] = df['valor_frete'] / df['valor_produtos'].replace(0, np.nan)

    # Verifica se o cliente realmente escreveu um comentário.
    mensagem = df['mensagem_comentario_avaliacao'].fillna('').astype(str).str.strip()
    tem_texto = mensagem.ne('') & mensagem.str.lower().ne('vazio')
    df['tem_comentario'] = tem_texto.astype(int)
    df['tamanho_comentario'] = mensagem.where(tem_texto, '').str.len()

    # Separa as notas baixas das altas para facilitar a comparação.
    df['nota_baixa'] = df['nota_avaliacao'].isin([1, 2]).astype(int)
    df['grupo_nota'] = np.select(
        [df['nota_avaliacao'].isin([1, 2]), df['nota_avaliacao'].eq(5)],
        ['baixa', 'alta'], default='intermediaria'
    )

    # Classifica as entregas conforme o prazo previsto.
    df['situacao_entrega'] = np.select(
        [df['desvio_prazo_dias'].lt(0), df['desvio_prazo_dias'].eq(0), df['desvio_prazo_dias'].gt(0)],
        ['antes_do_prazo', 'no_prazo', 'atrasada'], default='sem_data'
    )
    df['mes_compra'] = compra.dt.to_period('M').astype(str)
    return df

def transformar_dados(base):
    df = base.copy()

    # [IA] registro 3 - codificacao e escalas
    # Cada situação de entrega vira uma coluna com valores 0 ou 1.
    codificadas = pd.get_dummies(df['situacao_entrega'], prefix='entrega', dtype=int)
    df = pd.concat([df, codificadas], axis=1)

    # Aplica as duas escalas aos valores dos produtos, frete e tempo de entrega.
    for coluna in ['valor_produtos', 'valor_frete', 'tempo_entrega_dias']:
        valores = df[coluna]
        minimo, maximo = valores.min(), valores.max()
        media, desvio = valores.mean(), valores.std()
        df[coluna + '_minmax'] = (valores - minimo) / (maximo - minimo) if maximo > minimo else 0
        df[coluna + '_zscore'] = (valores - media) / desvio if desvio > 0 else 0
    return df

# [IA] registro N - Coloca a seleção existente da Frente 2 em uma função.
def selecionar_atributos(base):
    # Separa os atributos que serão analisados pela correlação e variância.
    colunas_numericas = [
        'qtd_itens', 'qtd_vendedores', 'qtd_avaliacoes', 'valor_produtos', 'valor_frete',
        'valor_total_pedido', 'proporcao_frete', 'tempo_entrega_dias',
        'tempo_ate_transportadora_dias', 'tempo_transporte_dias', 'desvio_prazo_dias',
        'dias_atraso', 'atrasou_entrega', 'tem_comentario', 'tamanho_comentario'
    ]
    colunas_codificadas = [col for col in base.columns if col.startswith('entrega_')]
    colunas_analisadas = colunas_numericas + colunas_codificadas

    # [IA] registro 4 - correlacao e baixa variancia
    # Verifica quais atributos têm relação com as notas e quais variam pouco.
    correlacoes = base[colunas_analisadas].corrwith(base['nota_avaliacao'])
    variancias = base[colunas_analisadas].var()
    selecao = pd.DataFrame({
        'atributo': colunas_analisadas,
        'correlacao_com_nota': correlacoes.reindex(colunas_analisadas).values,
        'variancia': variancias.reindex(colunas_analisadas).values
    })
    selecao['baixa_variancia'] = selecao['atributo'].isin(colunas_codificadas) & (selecao['variancia'] < 0.01)
    selecao['passa_correlacao'] = selecao['correlacao_com_nota'].abs() >= 0.05

    extras = selecao.loc[
        selecao['atributo'].isin(['qtd_vendedores', 'qtd_avaliacoes']) & selecao['passa_correlacao'],
        'atributo'
    ].tolist()
    extras += selecao.loc[
        selecao['atributo'].isin(colunas_codificadas) & ~selecao['baixa_variancia'],
        'atributo'
    ].tolist()

    # Mantém as colunas usadas na análise e nos gráficos do painel.
    colunas_finais = [
        'id_pedido', 'nota_avaliacao', 'grupo_nota', 'nota_baixa', 'mes_compra',
        'estado_cliente', 'categoria_produto', 'status_pedido', 'situacao_entrega',
        'valor_produtos', 'valor_frete', 'valor_total_pedido', 'proporcao_frete',
        'qtd_itens', 'tempo_entrega_dias',
        'tempo_ate_transportadora_dias', 'tempo_transporte_dias', 'desvio_prazo_dias',
        'dias_atraso', 'atrasou_entrega', 'tem_comentario', 'tamanho_comentario',
        'marca_temporal_compra_pedido', 'data_entrega_pedido_cliente', 'data_estimada_entrega_pedido'
    ]
    colunas_escalas = [c + '_' + tipo for c in ['valor_produtos', 'valor_frete', 'tempo_entrega_dias']
                    for tipo in ['minmax', 'zscore']]
    colunas_finais += colunas_escalas + extras
    analitica = base[colunas_finais].copy()

    # Confere se os pedidos são únicos, as notas são válidas e os cálculos estão ok.
    assert analitica['id_pedido'].is_unique
    assert analitica['nota_avaliacao'].between(1, 5).all()
    assert analitica['atrasou_entrega'].isna().equals(analitica['desvio_prazo_dias'].isna())
    assert analitica['valor_frete_minmax'].dropna().between(0, 1).all()


    return analitica, selecao

