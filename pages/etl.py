from encodings.punycode import T

import streamlit as st
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- CSS ---
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {display: none;}
    .stButton>button {
        background-color: #001F3F;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px;
        font-weight: 600;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.page_link("app.py", label="Início", icon="🏠")
    st.page_link("pages/analise_exploratoria.py", label="Análise Exploratória", icon="📄")
    st.page_link("pages/analise_grafica.py", label="Análise Gráfica", icon="📊")
    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    st.page_link("pages/resultados.py", label="Resultados", icon="🎯")

st.title("⚙️ Tratamento e Transformação de Dados")

# =====================================================
# VERIFICA DATAFRAME
# =====================================================

if "df" not in st.session_state:
    st.warning("Faça upload na Home")
    st.stop()

# SESSION STATE
if "df_temp" not in st.session_state:
    st.session_state["df_temp"] = st.session_state["df"].copy()

df_temp = st.session_state["df_temp"]

# =====================================================
# FUNÇÃO TIPO
# =====================================================

def get_tipo_icone(serie):
    if pd.api.types.is_integer_dtype(serie):
        return "🔢", "Inteiro"
    elif pd.api.types.is_float_dtype(serie):
        return "📊", "Float"
    elif pd.api.types.is_datetime64_any_dtype(serie):
        return "📅", "Data"
    elif pd.api.types.is_bool_dtype(serie):
        return "✔️", "Booleano"
    else:
        return "🔤", "Texto"

# =====================================================
# LOOP COLUNAS
# =====================================================

for coluna in df_temp.columns.tolist():

    st.markdown("---")

    icone, tipo = get_tipo_icone(df_temp[coluna])
    nulos = df_temp[coluna].isna().sum()
    unicos = df_temp[coluna].nunique()

    # HEADER
    col1, col2 = st.columns([20, 4])

    with col1:
        st.markdown(f"**{coluna}** {icone}  \n(tipo: {tipo} | nulos: {nulos} | únicos: {unicos})")

    with col2:
        if st.button("❌", key=f"del_{coluna}"):
            df_temp = df_temp.drop(columns=[coluna])
            st.session_state["df_temp"] = df_temp
            st.rerun()

    # =====================================================
    # LINHA SUPERIOR
    # =====================================================



    # GRAFICO

    if pd.api.types.is_numeric_dtype(df_temp[coluna]):

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

        fig.add_trace(go.Box(x=df_temp[coluna]), row=1, col=1)
        fig.add_trace(go.Histogram(x=df_temp[coluna]), row=2, col=1)

        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # ALTERAR TIPO
    
        

    # =====================================================
    # LINHA INFERIOR
    # =====================================================

    

    with st.container(border=True):
        col_r, col_n, col_o,col_tipo = st.columns(4)
        # REPLACE
        with col_r:
            novo_tipo = st.radio(
                "Tipo",
                ["Manter","int","float","str","datetime"],
                key=f"type_{coluna}"
            )

            if novo_tipo != "Manter":
                try:
                    if novo_tipo == "datetime":
                        df_temp[coluna] = pd.to_datetime(df_temp[coluna], errors="coerce")
                    else:
                        df_temp[coluna] = df_temp[coluna].astype(novo_tipo)
                except:
                    st.error("Erro ao converter")
        # NULOS
        with col_n:
            st.markdown("**Nulos**")

            opcao = st.radio(
                "Tratamento",
                ["Nenhum","Remover","Média","Mediana","Zero"],
                key=f"null_{coluna}"
            )

            if opcao == "Remover":
                df_temp = df_temp.dropna(subset=[coluna])
            elif opcao == "Média" and pd.api.types.is_numeric_dtype(df_temp[coluna]):
                df_temp[coluna] = df_temp[coluna].fillna(df_temp[coluna].mean())
            elif opcao == "Mediana" and pd.api.types.is_numeric_dtype(df_temp[coluna]):
                df_temp[coluna] = df_temp[coluna].fillna(df_temp[coluna].median())
            elif opcao == "Zero":
                df_temp[coluna] = df_temp[coluna].fillna(0)

        # OUTLIERS
        with col_o:
            if pd.api.types.is_numeric_dtype(df_temp[coluna]):

                st.markdown("**Outliers**")

                opcao = st.radio(
                    "Tratamento",
                    ["Nenhum","Remover","Limitar"],
                    key=f"out_{coluna}"
                )

                if opcao != "Nenhum":

                    Q1 = df_temp[coluna].quantile(0.25)
                    Q3 = df_temp[coluna].quantile(0.75)
                    IQR = Q3 - Q1

                    low = Q1 - 1.5 * IQR
                    high = Q3 + 1.5 * IQR

                    if opcao == "Remover":
                        df_temp = df_temp[(df_temp[coluna] >= low) & (df_temp[coluna] <= high)]
                    else:
                        df_temp[coluna] = np.clip(df_temp[coluna], low, high)
        with col_tipo:

                st.markdown("**Substituir**")

                old = st.text_input("Antigo", key=f"old_{coluna}")
                new = st.text_input("Novo", key=f"new_{coluna}")

                if old and new:
                    if df_temp[coluna].dtype == "object":
                        df_temp[coluna] = df_temp[coluna].str.replace(old, new, regex=False)
        # salvar alterações parciais
    st.session_state["df_temp"] = df_temp

# =====================================================
# FINAL
# =====================================================

st.divider()

if st.button("✅ Aplicar alterações"):
    st.session_state["df"] = df_temp.copy()
    st.success("Aplicado!")

st.dataframe(df_temp.head())

csv = df_temp.to_csv(index=False).encode("utf-8")

st.download_button("⬇️ Baixar CSV", csv, "dados.csv")