import streamlit as st
import pandas as pd

# ========================================
# CONFIGURAÇÃO DA PÁGINA
# ========================================
st.set_page_config(page_title="ABAK", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ========================================
# CSS CUSTOMIZADO
# ========================================
st.markdown("""
    <style>
    /* Cores principais */
    :root {
        --primary-color: #001F3F;
        --secondary-color: #10B981;
        --light-bg: #F3F4F6;
        --border-color: #D1D5DB;
    }
    
    /* Remove o menu padrão e sobe o conteúdo */
[data-testid="stSidebarNav"]: # "{ display: none; }"
[data-testid="stSidebarUserContent"]: # "{ padding-top: 1rem !important; }"

/* Estilo para os Links de Navegação (Substituindo os botões) */
[data-testid="stSidebar"]: # "a {"
    text-decoration: none !important;
    color: #001F3F !important; /* Azul Marinho */
    display: flex !important;
    align-items: center !important;
    padding: 10px 15px !important;
    border-radius: 10px !important;
    margin-bottom: 5px !important;
    transition: background-color 0.3s !important;
}

/* Efeito de Hover (Passar o mouse) */
[data-testid="stSidebar"]: # "a:hover {"
    background-color: #E5E7EB !important; /* Cinza claro */
}

/* CLASSE PARA PÁGINA ATIVA (Destaque Azul Claro) */
.active-nav {
    background-color: #E0F2FE !important; /* Azul bem clarinho */
    border-left: 5px solid #10B981 !important; /* Barra verde na lateral */
    font-weight: bold !important;
}

    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB;
        border-right: 2px solid var(--border-color);
    }
    
     /* Esconde a lista de páginas padrão do Streamlit na sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }

    /* Botões */
    .stButton>button {
        text-align: left !important;
        justify-content: flex-start !important;
        padding-left: 15px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        background-color: var(--primary-color);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #003366;
        box-shadow: 0 4px 12px rgba(0, 31, 63, 0.3);
    }

    /* Caixa de Upload */
    .upload-box {
        border: 2px dashed var(--border-color);
        border-radius: 20px;
        padding: 40px;
        text-align: center;
        background-color: #FAFBFC;
        margin: 20px 0;
    }

    .upload-icon {
        font-size: 48px;
        margin-bottom: 20px;
    }

    /* Caixa de Instruções */
    .instructions-box {
        background-color: #ECFDF5;
        border-left: 4px solid var(--secondary-color);
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }

    .instructions-box h4 {
        color: var(--secondary-color);
        margin-top: 0;
    }

    /* Quadro de Etapas */
    .steps-container {
        border: 2px solid var(--border-color);
        border-radius: 20px;
        padding: 40px 20px;
        background-color: white;
        margin: 30px 0;
    }

    .steps-title {
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 40px;
    }

    /* Círculos das Etapas */
    .step-circle {
        height: 60px;
        width: 60px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 24px;
        margin: 0 auto 15px;
    }

    .step-active {
        background-color: var(--secondary-color);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }

    .step-inactive {
        background-color: white;
        color: #9CA3AF;
        border: 2px solid var(--border-color);
    }

    .step-text {
        text-align: center;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 8px;
    }

    .step-description {
        text-align: center;
        font-size: 12px;
        color: #6B7280;
        line-height: 1.4;
    }

    /* Alerta */
    .alert-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        border-radius: 10px;
        padding: 15px;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ========================================
# SIDEBAR - LOGO E NAVEGAÇÃO
# ========================================
with st.sidebar:
    # Logo ABAK
    st.image("logo_abak_horizontal.png", use_container_width=True)
    st.markdown("---")

    # Navegação
    st.markdown("")

    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
    st.page_link("app.py", label="Início", icon="🏠")
    st.markdown('</div>', unsafe_allow_html=True)

    # As outras páginas ficam normais (sem a div active-nav)
    st.page_link("pages/analise_exploratoria.py", label="Análise Exploratória", icon="📄")
    st.page_link("pages/analise_grafica.py", label="Análise Gráfica", icon="📊")
    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    st.page_link("pages/resultados.py", label="Resultados e Predições", icon="🎯")


#    nav_options = {
#        "🏠 Início": "app.py",
#        "📄 Análise Exploratória": "pages/analise_exploratoria.py",
#        "📊 Análise Gráfica": "pages/analise_grafica.py",
#        "⚙️ Ferramentas ETL": "pages/etl.py",
#        "🎯 Resultados e Predições": "pages/resultados.py"
#    }
#
#    for label, page in nav_options.items():
#        if st.button(label, use_container_width=True, key=f"nav_{label}"):
#            st.switch_page(page)

# ========================================
# CONTEÚDO PRINCIPAL
# ========================================
st.title("🏠 Início")

# Caixa de Upload
st.markdown("""
    <style>
    /* 1. Aumenta e centraliza a zona de drop */
    div[data-testid="stFileUploader"] section {
        padding: 50px 20px !important;
        min-height: 300px !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }

    /* 2. Força o texto "Drag and drop..." a centralizar */
    div[data-testid="stFileUploader"] section div[role="button"] + div {
        text-align: center !important;
        display: block !important;
        width: 100%;
    }

    /* 4. Centraliza o botão Browse Files */
    div[data-testid="stFileUploader"] button {
        display: block;
        margin: 0 auto !important;
    }

    /* 5. Remove o estilo de grid que joga o texto para a esquerda */
    div[data-testid="stFileUploader"] section > div {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
# Instruções
st.info("Instruções para importação: O arquivo deve estar no formato CSV, XLSX ou JSON. A primeira linha deve conter os nomes das colunas.")
up_arquivo = st.file_uploader("", type=['csv', 'xlsx', 'json'])



# Processamento do arquivo
if up_arquivo is not None:
    try:
        # Detectar o tipo de arquivo
        if up_arquivo.name.endswith('.csv'):
            df = pd.read_csv(up_arquivo)
        elif up_arquivo.name.endswith('.xlsx'):
            df = pd.read_excel(up_arquivo)
        elif up_arquivo.name.endswith('.json'):
            df = pd.read_json(up_arquivo)

        # Salvar no session_state
        st.session_state['df'] = df
        st.success(f"✅ Arquivo '{up_arquivo.name}' carregado com sucesso! ({len(df)} linhas, {len(df.columns)} colunas)")
        st.write("### Pré Visualização (Primeiras 5 linhas)")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"❌ Erro ao carregar o arquivo: {e}")


# ========================================
# BOTÃO AVANÇAR
# ========================================
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])
with col3:
    if st.button("Avançar →", use_container_width=True):
        if "df" in st.session_state:
            st.switch_page("pages/analise_exploratoria.py")
        else:
            st.error("⚠️ Por favor, carregue um arquivo antes de prosseguir.")