import joblib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

async def criar_modelo():
    orders = pd.read_csv("dados/olist_orders_dataset.csv")
    customers = pd.read_csv("dados/olist_customers_dataset.csv")
    order_items = pd.read_csv("dados/olist_order_items_dataset.csv")
    products = pd.read_csv("dados/olist_products_dataset.csv")
    sellers = pd.read_csv("dados/olist_sellers_dataset.csv")
    geo = pd.read_csv("dados/olist_geolocation_dataset.csv")

    # Geo: média por prefixo
    geo['zip_prefix'] = geo['geolocation_zip_code_prefix']
    geo_grouped = geo.groupby('zip_prefix')[['geolocation_lat', 'geolocation_lng']].mean().reset_index()

    # Associações
    customers = customers.rename(columns={'customer_zip_code_prefix': 'zip_prefix'})
    sellers = sellers.rename(columns={'seller_zip_code_prefix': 'zip_prefix'})
    customers_geo = customers.merge(geo_grouped, on='zip_prefix', how='left')
    sellers_geo = sellers.merge(geo_grouped, on='zip_prefix', how='left')

    # Merge geral
    base = orders.merge(customers_geo, on='customer_id', how='left') \
                 .merge(order_items, on='order_id', how='left') \
                 .merge(products, on='product_id', how='left') \
                 .merge(sellers_geo, on='seller_id', how='left')

    base = base.rename(columns={
        'geolocation_lat_x': 'customer_lat',
        'geolocation_lng_x': 'customer_lng',
        'geolocation_lat_y': 'seller_lat',
        'geolocation_lng_y': 'seller_lng'
    })

    # Remove coordenadas inválidas
    base = base.dropna(subset=['customer_lat', 'customer_lng', 'seller_lat', 'seller_lng'])

    # Calcula distância
    base['distancia_km'] = base.apply(lambda row: geodesic(
        (row['customer_lat'], row['customer_lng']),
        (row['seller_lat'], row['seller_lng'])
    ).km, axis=1)

    # Target
    base['order_approved_at'] = pd.to_datetime(base['order_approved_at'])
    base['order_delivered_customer_date'] = pd.to_datetime(base['order_delivered_customer_date'])
    base['tempo_entrega_horas'] = (base['order_delivered_customer_date'] - base['order_approved_at']).dt.total_seconds() / 3600

    # Limpeza
    base = base.dropna(subset=['tempo_entrega_horas', 'price', 'freight_value'])
    base = base[base['tempo_entrega_horas'] > 0]

    # Features
    base['log_preco'] = np.log1p(base['price'])
    base['log_frete'] = np.log1p(base['freight_value'])
    base['frete_pct'] = base['freight_value'] / base['price']

    # Final: features e target
    features = ['log_preco', 'log_frete', 'distancia_km', 'frete_pct']
    target = 'tempo_entrega_horas'

    X = base[features]
    y = base[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, random_state=42)

    # Modelo: RandomForest otimizado
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Métricas
    st.metric("🎯 R² (precisão)", f"{r2_score(y_test, y_pred):.2f}")
    st.metric("📉 Erro Médio Absoluto", f"{mean_absolute_error(y_test, y_pred):.2f} horas")
    st.metric("📊 RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f} horas")

    joblib.dump({
    'model': model,
    'scaler': scaler,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test
    }, 'yuri_teste.joblib')
