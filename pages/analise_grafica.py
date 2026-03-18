import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================
# CSS
# ==========================

st.markdown("""
<style>
[data-testid="stSidebarNav"] {display: none;}
[data-testid="stSidebarUserContent"] {padding-top: 1rem !important;}
[data-testid="stSidebar"] img {margin-top: -30px; margin-bottom: -10px;}
[data-testid="stSidebar"] a {
    text-decoration: none !important;
    color: #001F3F !important;
    display: flex !important;
    align-items: center !important;
    padding: 10px 15px !important;
    border-radius: 10px !important;
    margin-bottom: 5px !important;
}
[data-testid="stSidebar"] a:hover {background-color: #E5E7EB !important;}
.active-nav {
    background-color: #E0F2FE !important;
    border-left: 5px solid #10B981 !important;
    font-weight: bold !important;
}
.stButton>button {
    background-color: #001F3F;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px 24px;
    font-weight: 600;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.image("logo_abak_horizontal.png", use_container_width=True)
    st.markdown("---")

    st.page_link("app.py", label="Início", icon="🏠")
    st.page_link("pages/analise_exploratoria.py", label="Análise Exploratória", icon="📄")

    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
    st.page_link("pages/analise_grafica.py", label="Análise Gráfica", icon="📊")
    st.markdown('</div>', unsafe_allow_html=True)

    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    st.page_link("pages/resultados.py", label="Resultados e Predições", icon="🎯")

st.title("📊 Análise Gráfica")

# ==========================
# FUNÇÕES
# ==========================

@st.cache_data
def gerar_matriz_correlacao(df_base: pd.DataFrame):
    return df_base.corr(numeric_only=True)

# ==========================
# APP
# ==========================

if "df" in st.session_state:

    dataframe_original = st.session_state["df"].copy()
    dataframe_original.columns = dataframe_original.columns.str.strip()

    # Converter números com vírgula
    for coluna in dataframe_original.columns:
        try:
            dataframe_original[coluna] = pd.to_numeric(
                dataframe_original[coluna].astype(str).str.replace(",", ".")
            )
        except (ValueError, TypeError):
            pass

    if len(dataframe_original) > 100000:
        dataframe_original = dataframe_original.sample(100000, random_state=42)

    dataframe_filtrado = dataframe_original.copy()

    # ==========================
    # FILTROS E GRÁFICOS JUNTOS
    # ==========================

    col_filtros, col_graficos = st.columns([1,3])

    with col_filtros:

        st.subheader("🔎 Filtros")

        colunas_categoricas = (
            dataframe_filtrado.select_dtypes(include="object").columns.tolist()
        )

        colunas_numericas = (
            dataframe_filtrado.select_dtypes(include=np.number).columns.tolist()
        )

        mascara_filtro = pd.Series(True, index=dataframe_filtrado.index)

        # filtros categóricos
        for coluna_cat in colunas_categoricas:
            valores_unicos = dataframe_original[coluna_cat].dropna().unique()
            valores_selecionados = st.multiselect(
                f"Filtrar {coluna_cat}",
                options=valores_unicos
            )
            if valores_selecionados:
                mascara_filtro &= dataframe_filtrado[coluna_cat].isin(valores_selecionados)

        # filtros numéricos
        for coluna_num in colunas_numericas:
            minimo_valor = float(dataframe_original[coluna_num].min())
            maximo_valor = float(dataframe_original[coluna_num].max())
            intervalo = st.slider(
                coluna_num,
                minimo_valor,
                maximo_valor,
                (minimo_valor, maximo_valor)
            )
            mascara_filtro &= (
                (dataframe_filtrado[coluna_num] >= intervalo[0]) &
                (dataframe_filtrado[coluna_num] <= intervalo[1])
            )

        dataframe_filtrado = dataframe_filtrado[mascara_filtro]

    # ==========================
    # GRÁFICOS
    # ==========================

    with col_graficos:

        st.subheader("📈 Análise Gráfica")

        lista_colunas = dataframe_filtrado.columns.tolist()

        tipo_grafico = st.pills(
            "Escolha o tipo de gráfico",
            ["Dispersão", "Linhas", "Histograma", "Boxplot", "Heatmap"]
        )

        coluna_x = st.selectbox("Coluna X", lista_colunas)
        coluna_y = st.selectbox(
            "Coluna Y (opcional)",
            options=[None] + lista_colunas,
            format_func=lambda v: "Nenhuma" if v is None else v
        )
        coluna_cor = st.selectbox(
            "Segmentar por cor",
            options=[None] + lista_colunas,
            format_func=lambda v: "Nenhuma" if v is None else v
        )

        figura_final = None

        if tipo_grafico == "Dispersão" and coluna_y:
            df_plot = dataframe_filtrado.copy()
            df_plot[coluna_x] = pd.to_numeric(df_plot[coluna_x], errors="coerce")
            df_plot[coluna_y] = pd.to_numeric(df_plot[coluna_y], errors="coerce")
            df_plot = df_plot.dropna(subset=[coluna_x, coluna_y])
            figura_final = px.scatter(
                df_plot,
                x=coluna_x,
                y=coluna_y,
                color=coluna_cor,
                trendline="ols"
            )

        elif tipo_grafico == "Linhas" and coluna_y:
            df_plot = dataframe_filtrado.copy()
            try:
                df_plot[coluna_x] = pd.to_datetime(df_plot[coluna_x])
            except (ValueError, TypeError):
                pass
            figura_final = px.line(
                df_plot.sort_values(by=coluna_x),
                x=coluna_x,
                y=coluna_y,
                color=coluna_cor
            )

        elif tipo_grafico == "Histograma":
            figura_final = px.histogram(
                dataframe_filtrado,
                x=coluna_x,
                color=coluna_cor,
                nbins=30
            )

        elif tipo_grafico == "Boxplot" and coluna_y:
            df_plot = dataframe_filtrado.copy()
            df_plot[coluna_y] = pd.to_numeric(df_plot[coluna_y], errors="coerce")
            df_plot = df_plot.dropna(subset=[coluna_y])
            figura_final = px.box(
                df_plot,
                x=coluna_x,
                y=coluna_y,
                color=coluna_cor
            )

        elif tipo_grafico == "Heatmap":
            matriz_corr = gerar_matriz_correlacao(dataframe_filtrado)
            if not matriz_corr.empty:
                figura_final = px.imshow(
                    matriz_corr,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="RdBu_r"
                )

        if figura_final:
            st.plotly_chart(figura_final, use_container_width=True)
        else:
            st.warning("Selecione colunas válidas.")


    # ==========================
    # TABELA FINAL
    # ==========================

    st.subheader("📄 Dados Filtrados")
    st.dataframe(dataframe_filtrado)

else:
    st.info("Envie um arquivo CSV para começar.")

# ==========================
# BOTÕES DE NAVEGAÇÃO
# ==========================

col_v, col_espaco, col_a = st.columns([1,3,1])

with col_v:
    if st.button("← Voltar"):
        st.switch_page("pages/analise_exploratoria.py")

with col_a:
    if st.button("Avançar →"):
        st.switch_page("pages/etl.py")