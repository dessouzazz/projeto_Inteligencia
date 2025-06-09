import asyncio
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import plotly.express as px
import lightgbm as lgb
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from engine import criar_modelo

# Carregar os dados
sellers = pd.read_csv("dados/olist_sellers_dataset.csv")
geo = pd.read_csv("dados/olist_geolocation_dataset.csv")


    
# Configuração inicial da página
st.set_page_config(page_title="Dashboard Olist", page_icon="📦", layout="wide")

# Base path
BASE_PATH = 'dados/'

# Cache para carregamento de dados
@st.cache_data
def load_data():
    orders = pd.read_csv(BASE_PATH + 'olist_orders_dataset.csv')
    customers = pd.read_csv(BASE_PATH + 'olist_customers_dataset.csv')
    reviews = pd.read_csv(BASE_PATH + 'olist_order_reviews_dataset.csv')
    geolocation = pd.read_csv(BASE_PATH + 'olist_geolocation_dataset.csv')
    order_items = pd.read_csv(BASE_PATH + 'olist_order_items_dataset.csv')
    products = pd.read_csv(BASE_PATH + 'olist_products_dataset.csv')
    categories = pd.read_csv(BASE_PATH + 'product_category_name_translation.csv')
    return orders, customers, reviews, geolocation, order_items, products, categories

# Carrega todos os datasets
orders, customers, reviews, geolocation, order_items, products, categories = load_data()

# Juntando pedidos com clientes
orders_customers = pd.merge(orders, customers, on='customer_id', how='inner')

# Título do App
st.title("📊 Predição de Categoria de Produto no Varejo")

#### GERAR UM NOVO MODELO DE FORMA ASSINCRONA. 
if not os.path.exists('yuri_teste.joblib'):
    st.warning("🔄 Modelo ainda não treinado. Treinando agora...")
    
    # Executa função assíncrona de forma segura
    asyncio.run(criar_modelo())
# Upload de CSV
uploaded_file = st.sidebar.file_uploader("📂 Faça upload de um CSV de produtos:", type="csv")
if uploaded_file:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        st.subheader("Visualização da Base Carregada")
        st.dataframe(df_uploaded.head(10))
        st.markdown("---")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

# Menu de navegação
menu = st.sidebar.selectbox("Escolha uma visualização:", [
    "Pedidos por Estado", 
    "Avaliações por Categoria",
    "Mapa de Clientes",
    "Previsão de Entrega (IA)",
    "Análise Exploratória",
    "Dashboard"
])

# ======== VISUALIZAÇÃO 1: Pedidos por Estado ========
if menu == "Pedidos por Estado":
    pedidos_estado = orders_customers['customer_state'].value_counts().reset_index()
    pedidos_estado.columns = ['Estado', 'Total de Pedidos']

    fig = px.bar(pedidos_estado, x='Estado', y='Total de Pedidos',
                 color='Estado', title='Total de Pedidos por Estado')
    st.plotly_chart(fig)

# ======== VISUALIZAÇÃO 2: Avaliações por Categoria ========
elif menu == "Avaliações por Categoria":
    merged = reviews.merge(orders, on='order_id', how='inner') \
                    .merge(order_items, on='order_id', how='inner') \
                    .merge(products, on='product_id', how='inner') \
                    .merge(categories, on='product_category_name', how='left')

    nota_categoria = merged.groupby('product_category_name_english')['review_score'] \
                           .mean().sort_values(ascending=False).head(15)

    import plotly.express as px

    fig = px.bar(
    nota_categoria.sort_values(),  # Garantir ordem crescente
    x=nota_categoria.sort_values().values,
    y=nota_categoria.sort_values().index,
    orientation='h',
    text=nota_categoria.sort_values().values,
    labels={'x': 'Nota Média', 'y': 'Categoria'},
    color_discrete_sequence=['#03A9F4'],  # Azul neon mais vibrante
    title='📊 Média das Avaliações por Categoria'
    )

    fig.update_traces(
    texttemplate='%{text:.2f}',
    textposition='outside',
    marker_line_color='white',
    marker_line_width=1.2
    )

    fig.update_layout(
    plot_bgcolor='#0d1117',  # fundo escuro elegante
    paper_bgcolor='#0d1117',
    font=dict(color='white', size=15),
    title_font=dict(size=22, color='white', family='Arial'),
    xaxis=dict(
        showgrid=True,
        gridcolor='#333333',
        zeroline=False,
        color='white'
        ),
    yaxis=dict(
        showgrid=False,
        color='white',
        tickfont=dict(size=13),
        ),
    margin=dict(l=180, r=50, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)

# ======== VISUALIZAÇÃO 3: Mapa de Clientes ========
elif menu == "Mapa de Clientes":
    estados_selecionados = st.multiselect(
        'Selecione os estados:',
        options=sorted(geolocation['geolocation_state'].dropna().unique()),
        default=['SP', 'RJ']
    )
    mapa = geolocation[geolocation['geolocation_state'].isin(estados_selecionados)]

    mapa_renomeado = mapa.rename(columns={
        'geolocation_lat': 'latitude',
        'geolocation_lng': 'longitude'
    })

    fig = px.scatter_mapbox(
        mapa_renomeado,
        lat='latitude',
        lon='longitude',
        zoom=4,
        height=600,
        opacity=0.5,
        title='Distribuição Geográfica dos Clientes',
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig)

# ======== VISUALIZAÇÃO 4: Previsão de Entrega com IA ========
elif menu == "Previsão de Entrega (IA)":
    st.subheader("🔮 Previsão Inteligente do Tempo de Entrega")

    modelo_salvo = joblib.load("yuri_teste.joblib")
    model = modelo_salvo['model']
    scaler = modelo_salvo['scaler']
    X_train = modelo_salvo['X_train']
    X_test = modelo_salvo['X_test']
    y_train = modelo_salvo['y_train']
    y_test = modelo_salvo['y_test']
    y_pred = model.predict(X_test)


    # Interface de Previsão
    st.markdown("### ✏️ Faça sua própria previsão:")

    preco_input = st.number_input("Preço do produto (R$):", 10.0, 10000.0, 100.0, step=10.0)
    frete_input = st.number_input("Valor do frete (R$):", 0.0, 1000.0, 20.0, step=5.0)
    distancia_input = st.number_input("Distância estimada entre cliente e vendedor (km):", 0.0, 3000.0, 100.0, step=10.0)

    if st.button("🚚 Prever tempo de entrega"):
        entrada = pd.DataFrame([{
            'log_preco': np.log1p(preco_input),
            'log_frete': np.log1p(frete_input),
            'distancia_km': distancia_input,
            'frete_pct': frete_input / preco_input
        }])

        entrada_scaled = scaler.transform(entrada)
        pred = model.predict(entrada_scaled)

        # Métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 R² Score", f"{r2:.2%}")
        col2.metric("📉 MAE", f"{mae:.2f} horas")
        col3.metric("📏 RMSE", f"{rmse:.2f} horas")
        st.success(f"⏱️ Tempo estimado de entrega: {int(np.round(pred[0]))} horas")

# ======== VISUALIZAÇÃO 5: Análise Exploratória ========
elif menu == "Análise Exploratória":
    st.subheader("🔍 Análise Exploratória de Produtos")

    merged = order_items.merge(products, on='product_id', how='inner')
    merged = merged.merge(categories, on='product_category_name', how='left')

    vendas_categoria = merged.groupby('product_category_name_english')['price'].sum() \
        .sort_values(ascending=False).head(10)

    fig1 = px.bar(
        vendas_categoria,
        x=vendas_categoria.index,
        y=vendas_categoria.values,
        title='Top 10 Categorias por Faturamento',
        labels={'x': 'Categoria', 'y': 'Faturamento (R$)'}
    )
    st.plotly_chart(fig1)

    media_frete_categoria = merged.groupby('product_category_name_english')['freight_value'].mean() \
        .sort_values(ascending=False).head(10)

    fig2 = px.bar(
        media_frete_categoria,
        x=media_frete_categoria.index,
        y=media_frete_categoria.values,
        title='Top 10 Categorias por Custo Médio de Frete',
        labels={'x': 'Categoria', 'y': 'Frete Médio (R$)'}
    )
    st.plotly_chart(fig2)

    st.markdown("---")
    st.write("### Visualização Completa dos Dados de Produtos")
    st.dataframe(merged.head(100))


# ======== Dashboard de Métricas ========
elif menu == "📈 Dashboard de Métricas e Previsões":
    st.header("📈 Dashboard de Métricas e Previsões do Modelo IA")

    try:
        modelo_salvo = joblib.load("yuri_teste.joblib")
        model = modelo_salvo['model']
        scaler = modelo_salvo['scaler']
        X_train = modelo_salvo['X_train']
        X_test = modelo_salvo['X_test']
        y_train = modelo_salvo['y_train']
        y_test = modelo_salvo['y_test']
        y_pred = model.predict(X_test)

        # Métricas
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        col1, col2, col3 = st.columns(3)
        col1.metric("🎯 R² Score", f"{r2:.2%}")
        col2.metric("📉 MAE", f"{mae:.2f} horas")
        col3.metric("📏 RMSE", f"{rmse:.2f} horas")

        # Histograma de previsões
        fig_hist = px.histogram(y_pred, nbins=30, title="Distribuição das Previsões de Entrega (em horas)")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Gráfico real vs previsto
        fig_scatter = px.scatter(
            x=y_test, 
            y=y_pred,
            labels={'x': 'Tempo Real (h)', 'y': 'Tempo Previsto (h)'},
            title="🔍 Comparativo: Tempo Real vs Previsto"
        )
        fig_scatter.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                              line=dict(color="red", dash="dash"))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Previsão personalizada
        st.markdown("---")
        st.subheader("🔢 Faça uma nova previsão manual:")

        preco_input = st.number_input("Preço do produto (R$):", 10.0, 10000.0, 100.0, step=10.0)
        frete_input = st.number_input("Valor do frete (R$):", 0.0, 1000.0, 20.0, step=5.0)
        distancia_input = st.number_input("Distância estimada entre cliente e vendedor (km):", 0.0, 3000.0, 100.0, step=10.0)

        if st.button("📦 Prever Tempo de Entrega"):
            entrada = pd.DataFrame([{
                'log_preco': np.log1p(preco_input),
                'log_frete': np.log1p(frete_input),
                'distancia_km': distancia_input,
                'frete_pct': frete_input / preco_input
            }])
            entrada_scaled = scaler.transform(entrada)
            pred_novo = model.predict(entrada_scaled)[0]
            print('Previsao - > ', pred_novo)
            st.success(f"Previsão: {pred_novo}")
            st.success(f"⏱️ Tempo estimado de entrega: **{int(np.round(pred_novo))} horas**")

    except Exception as e:
        st.error(f"Erro ao carregar modelo ou gerar métricas: {e}")

