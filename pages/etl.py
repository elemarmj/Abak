import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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
    st.page_link("pages/analise_exploratoria.py", label="Análise Exploratória", icon="📄")
    st.page_link("pages/analise_grafica.py", label="Análise Gráfica", icon="📊")

    # Destaque ATIVO para esta página (ETL)
    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    st.markdown('</div>', unsafe_allow_html=True)

    st.page_link("pages/resultados.py", label="Resultados e Predições", icon="🎯")
# --- FIM DA SIDEBAR ---


st.title("⚙️ Tratamento e Transformação de Dados")

# =====================================================
# VERIFICA SE DATAFRAME EXISTE
# =====================================================

if "df" not in st.session_state:
    st.warning("Por favor, faça upload do arquivo na página Home.")
    st.stop()

df_original = st.session_state["df"]
df_temp = df_original.copy()

# =====================================================
# FUNÇÃO TIPO + ÍCONE
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


st.divider()

# =====================================================
# LOOP POR COLUNA
# =====================================================

for coluna in df_original.columns.tolist():

    if coluna not in df_temp.columns:
        continue

    st.markdown("---")

    icone, tipo = get_tipo_icone(df_temp[coluna])

    nulos = df_temp[coluna].isna().sum()
    unicos = df_temp[coluna].nunique()

    # =====================================================
    # HEADER
    # =====================================================

    col_header, col_delete = st.columns([20,4])

    with col_header:

        st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:6px;">
        <span style="font-size:18px;">🔹</span>
        <span style="font-size:18px; font-weight:600;">{coluna}</span>
        <span style="font-size:16px;">{icone}</span>
        <span style="color:gray; font-size:13px;">
        ({tipo} | Nulos: {nulos} | Únicos: {unicos})
        </span>
        </div>
        """,
        unsafe_allow_html=True
        )

    with col_delete:

        if st.button("❌ Excluir coluna", key=f"delete_{coluna}"):

            df_temp = df_temp.drop(columns=[coluna])

            st.session_state["df"] = df_temp.copy()

            st.rerun()

    # =====================================================
    # LAYOUT PRINCIPAL
    # =====================================================

    col_tratamento, col_grafico = st.columns([1,1])

    # =====================================================
    # TRATAMENTOS
    # =====================================================

    with col_tratamento:

        # Alterar tipo
        novo_tipo = st.radio(
            "Alterar tipo",
            ["Manter","int","float","str","datetime"],
            key=f"type_{coluna}"
        )

        if novo_tipo != "Manter":

            try:

                if novo_tipo == "datetime":

                    df_temp[coluna] = pd.to_datetime(
                        df_temp[coluna],
                        errors="coerce"
                    )

                else:

                    df_temp[coluna] = df_temp[coluna].astype(novo_tipo)

            except:

                st.error("Erro ao converter tipo")


        # Replace
        st.markdown("### Substituir valor")

        valor_antigo = st.text_input(
            "Valor antigo",
            key=f"old_{coluna}"
        )

        valor_novo = st.text_input(
            "Novo valor",
            key=f"new_{coluna}"
        )

        if valor_antigo and valor_novo:

            df_temp[coluna] = df_temp[coluna].astype(str).str.replace(
                valor_antigo,
                valor_novo,
                regex=False
            )


        # Nulos
        opcao_nulo = st.radio(
            "Tratar nulos",
            [
            "Nenhum",
            "Remover linhas",
            "Preencher média",
            "Preencher mediana",
            "Preencher 0"
            ],
            key=f"null_{coluna}"
        )

        if opcao_nulo == "Remover linhas":

            df_temp = df_temp.dropna(subset=[coluna])

        elif opcao_nulo == "Preencher média":

            if pd.api.types.is_numeric_dtype(df_temp[coluna]):

                df_temp[coluna] = df_temp[coluna].fillna(
                    df_temp[coluna].mean()
                )

        elif opcao_nulo == "Preencher mediana":

            if pd.api.types.is_numeric_dtype(df_temp[coluna]):

                df_temp[coluna] = df_temp[coluna].fillna(
                    df_temp[coluna].median()
                )

        elif opcao_nulo == "Preencher 0":

            df_temp[coluna] = df_temp[coluna].fillna(0)


        # Outliers
        if pd.api.types.is_numeric_dtype(df_temp[coluna]):

            opcao_outlier = st.radio(
                "Outliers",
                [
                "Nenhum",
                "Remover (IQR)",
                "Limitar (Cap IQR)"
                ],
                key=f"out_{coluna}"
            )

            if opcao_outlier != "Nenhum":

                Q1 = df_temp[coluna].quantile(0.25)
                Q3 = df_temp[coluna].quantile(0.75)

                IQR = Q3 - Q1

                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR

                if opcao_outlier == "Remover (IQR)":

                    df_temp = df_temp[
                        (df_temp[coluna] >= lower) &
                        (df_temp[coluna] <= upper)
                    ]

                elif opcao_outlier == "Limitar (Cap IQR)":

                    df_temp[coluna] = np.where(
                        df_temp[coluna] < lower,
                        lower,
                        np.where(
                            df_temp[coluna] > upper,
                            upper,
                            df_temp[coluna]
                        )
                    )

    # =====================================================
    # GRAFICO
    # =====================================================

    with col_grafico:

        if pd.api.types.is_numeric_dtype(df_temp[coluna]):

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.25,0.75]
            )

            fig.add_trace(
                go.Box(
                    x=df_temp[coluna],
                    boxpoints="outliers"
                ),
                row=1,
                col=1
            )

            fig.add_trace(
                go.Histogram(
                    x=df_temp[coluna],
                    nbinsx=30
                ),
                row=2,
                col=1
            )

            fig.update_layout(
                height=400,
                showlegend=False
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =====================================================
# BOTÃO APLICAR ALTERAÇÕES
# =====================================================

st.divider()

if st.button(
    "✅ Aplicar alterações",
    use_container_width=True
):

    st.session_state["df"] = df_temp.copy()

    st.success("Alterações aplicadas com sucesso!")


# =====================================================
# VISUALIZAÇÃO
# =====================================================

st.subheader("📊 Pré-visualização")

st.dataframe(df_temp.sample(5))


csv = df_temp.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "⬇️ Baixar CSV tratado",
    csv,
    "dados_tratados.csv",
    "text/csv",
    use_container_width=True
)

# --- INÍCIO DOS BOTÕES INFERIORES ---
st.markdown("", unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns(5)

with col2:
    if st.button("← ← Voltar (Análise Explor.)"):
        st.switch_page("pages/analise_exploratoria.py")

with col3:
    if st.button("← Voltar (Análise Gráf.)"):
        st.switch_page("pages/analise_grafica.py")

with col5:
    if st.button("Avançar →"):
        st.switch_page("pages/resultados.py")
# --- FIM DOS BOTÕES INFERIORES ---