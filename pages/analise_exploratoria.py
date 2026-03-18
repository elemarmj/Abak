import streamlit as st
import pandas as pd

# --- INÍCIO DO BLOCO CSS ---
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
        transition: background-color 0.3s !important;
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
# --- FIM DO BLOCO CSS ---


# --- INÍCIO DA SIDEBAR ---
with st.sidebar:
    st.image("logo_abak_horizontal.png", use_container_width=True)
    st.markdown("---")

    st.page_link("app.py", label="Início", icon="🏠")

    # Destaque ATIVO para esta página (Análise Exploratória)
    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
    st.page_link("pages/analise_exploratoria.py", label="Análise Exploratória", icon="📄")
    st.markdown('</div>', unsafe_allow_html=True)

    st.page_link("pages/analise_grafica.py", label="Análise Gráfica", icon="📊")
    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    st.page_link("pages/resultados.py", label="Resultados e Predições", icon="🎯")
# --- FIM DA SIDEBAR ---


st.title("📄 Análise Exploratória")

if "df" not in st.session_state:
    st.warning("Por favor, faça upload do arquivo na página Home.")
    st.stop()

df = st.session_state["df"]

st.subheader("Informações gerais")
st.write(df.shape)
st.write(df.describe())

st.subheader("Valores nulos")
nulos = df.isnull().sum().reset_index()
nulos.columns = ['Coluna', 'Total Nulos']
st.write(nulos)

st.session_state["df"] = df

# --- INÍCIO DOS BOTÕES INFERIORES ---
st.markdown("", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col2:
    if st.button("← Voltar"):
        st.switch_page("app.py")

with col4:
    if st.button("Avançar para ETL → →"):
        st.switch_page("pages/etl.py")

with col5:
    if st.button("Avançar para Análise Gráfica →"):
        st.switch_page("pages/analise_grafica.py")
# --- FIM DOS BOTÕES INFERIORES ---