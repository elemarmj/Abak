# Imports
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import pickle
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    st.page_link("pages/etl.py", label="Ferramentas ETL", icon="⚙️")
    # Destaque ATIVO para esta página (Análise Exploratória)
    st.markdown('<div class="active-nav">', unsafe_allow_html=True)
    st.page_link("pages/resultados.py", label="Resultados e Predições", icon="🎯")
    st.markdown('</div>', unsafe_allow_html=True)
# --- FIM DA SIDEBAR ---





# Configuração da Página
st.set_page_config(page_title="Abak - Resultados", layout="wide")
st.title("Resultados e Predição")

# Verificação de segurança: Se não houver dados, para a execução
if "df" not in st.session_state:
    st.warning("⚠️ Por favor, faça upload do arquivo na página Home antes de prosseguir.")
    if st.button("Ir para Home"):
        st.switch_page("app.py")
    st.stop()

df = st.session_state["df"]

# ==================================================
# PASSO 1: Configuração do Treinamento
# ==================================================
st.subheader("1️⃣ Configuração do Modelo")
coluna_target = st.selectbox("Selecione a coluna Target (Alvo):", df.columns,index=None,placeholder="Escolha uma coluna...")

# Se o usuário mudar o target ou ainda não selecionou nada, limpamos as travas
if "target_anterior" not in st.session_state or st.session_state["target_anterior"] != coluna_target:
    st.session_state['treino_concluido'] = False
    st.session_state['mostrar_simulador'] = False
    st.session_state["target_anterior"] = coluna_target

# Trava 1: Se não escolheu o target, para aqui.
if not coluna_target:
    st.info("👆 Selecione uma coluna alvo para habilitar o treinamento.")
    st.session_state['treino_concluido'] = False
    st.stop()

col_voltar, col_treinar = st.columns([1, 5])
with col_voltar:
    if st.button("← Voltar"):
        st.switch_page("pages/etl.py")

with col_treinar:
    if st.button("🎯 Treinar Modelos"):
        with st.spinner("Processando dados e treinando os 5 modelos..."):
            try:
                # Separação entre Previsores e Classe
                X = df.drop(coluna_target, axis=1).values
                y = df[coluna_target].values

                # Escalonamento de valores
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                st.session_state["scaler_objeto"] = scaler
                # Separação entre treino e teste
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Dicionário de Modelos
                modelos = {'Naive Bayes': GaussianNB(),
                           'Decision Tree': DecisionTreeClassifier(),
                           'KNN': KNeighborsClassifier(5),
                           'Random Forest': RandomForestClassifier(100),
                           'SVM': SVC()}

                resultados_lista = []
                modelos_objetos = {}  # Para guardar os modelos e usar na predição depois

                # Loop de Treinamento
                for nome, modelo in modelos.items():
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)

                    #Adiciona os resultados na lista
                    resultados_lista.append({
                        "Modelo": nome,
                        "Acurácia": acc,
                        "Acurácia %": f"{acc:.2%}"
                    })
                    modelos_objetos[nome] = modelo

                #Salvando resultados no Session State
                st.session_state["df_resultados"] = pd.DataFrame(resultados_lista).sort_values(by="Acurácia", ascending=False)
                st.session_state["modelos_objetos"] = modelos_objetos
                st.session_state["target_selecionado"] = coluna_target
                st.session_state["treino_concluido"] = True
                st.session_state['X_test'] = X_test
                st.session_state['y_test'] = y_test

                st.success("✅ Modelos treinados com sucesso!")
                st.rerun()  # Recarrega para mostrar o Passo 2
            except Exception as e:
                st.error(f"Erro ao treinar modelos: {e}. Verifique se todas as colunas são numéricas após o ETL.")

# ==================================================
# PASSO 2: Comparação e Escolha do Modelo
# ==================================================
if st.session_state.get("treino_concluido", False):
    st.markdown("---")
    st.subheader("2️⃣ Comparação de Performance")

    with st.expander("❓ Como ler este relatório? (Guia Rápido)"):
        st.markdown("""
        O **Relatório de Classificação** mostra como o modelo se comporta para cada categoria:

        **Métricas por Categoria:**
        *   **Precision (Precisão)**: De todos que o modelo classificou como 'X', quantos eram realmente 'X'? (Evita "alarmes falsos").
        *   **Recall (Sensibilidade)**: De todos os 'X' que existem de verdade, quantos o modelo conseguiu encontrar? (Evita "esquecimentos").
        *   **F1-Score**: É a média entre Precisão e Sensibilidade. O valor ideal é **1.00**.
        *   **Support**: Quantos exemplos reais daquela categoria foram usados no teste.
        
        **Métricas Globais (Resumo):**
        *   **Accuracy (Acurácia)**: A porcentagem total de acertos do modelo (geral).
        *   **Macro Avg (Média Simples)**: Calcula a média das métricas sem considerar se uma categoria tem mais exemplos que outra. Trata todas as categorias com o mesmo peso.
        *   **Weighted Avg (Média Ponderada)**: Calcula a média levando em conta o tamanho de cada categoria (**Support**). Se uma categoria aparece muito mais, ela terá mais peso no resultado final.
        
        > **Dica**: Se a **Macro Avg** for muito menor que a **Weighted Avg**, significa que o modelo é ruim para categorias que aparecem pouco nos seus dados.
        """)

    # Recuperamos os dados necessários do session_state
    df_resultados = st.session_state['df_resultados']
    modelos_objetos = st.session_state['modelos_objetos']
    X_test = st.session_state['X_test']
    y_test = st.session_state['y_test']
    # Ordena o df para colocar a maior acurácio primeiro
    df_ordenado = df_resultados.sort_values(by="Acurácia", ascending=False)

    # Percorremos cada modelo treinado para criar seu bloco exclusivo
    for i, (index_original, row) in enumerate(df_ordenado.iterrows()):
        nome_modelo = row['Modelo']
        with st.container():
            st.markdown(f"### 🤖 Modelo: {nome_modelo} (Rank #{i + 1})")

            # 1. Pegamos a linha correspondente na tabela de resultados
            dados_modelo = df_resultados[df_resultados['Modelo'] == nome_modelo]

            # 2. Exibimos a tabela resumida com barras de progresso
            st.dataframe(
                dados_modelo,
                column_config={
                    "Acurácia": st.column_config.ProgressColumn(
                        "Acurácia", format="%.2f", min_value=0, max_value=1, color="green"
                    ),
                    "F1-Score": st.column_config.ProgressColumn(
                        "F1-Score", format="%.2f", min_value=0, max_value=1, color="blue"
                    ),
                    "Acurácia %": "Taxa de Acerto",
                    "F1-Score %": "Equilíbrio (F1)"
                },
                hide_index=True,
                use_container_width=True
            )

            # 3. Geramos e exibimos o Classification Report (O "pulo do gato")
            st.markdown("**Relatório de Classificação Detalhado:**")
            modelo_atual = modelos_objetos[nome_modelo]
            y_pred = modelo_atual.predict(X_test)

            # O report é gerado como texto formatado
            report = classification_report(y_test, y_pred)

            # Usamos st.code para manter o alinhamento das colunas do relatório
            st.code(report, language="text")

            # Separador visual entre os blocos de modelos
            st.divider()

    lista_modelos_ordenada = df_ordenado['Modelo'].tolist()

    modelo_escolhido = st.selectbox(
        "Escolha o modelo que deseja usar para a simulação:",
        lista_modelos_ordenada,
        index=None,
        placeholder="Selecione o algoritmo..."
    )

# Trava: Só mostra o botão de Iniciar Simulador se um modelo for escolhido
    if not modelo_escolhido:
        st.warning("👈 Selecione um modelo na lista acima para prosseguir.")
    else:
        st.session_state['modelo_selecionado_nome'] = modelo_escolhido
        st.info(f"Modelo **{modelo_escolhido}** pronto para simulação.")
        colA, colB = st.columns(2)
        with colA:
            if st.button("🚀 Iniciar Simulador"):
                st.session_state['mostrar_simulador'] = True
                st.rerun()
        with colB:
            # 1. Preparar o modelo em memória (sem precisar salvar no disco)
            buffer = io.BytesIO()
            pickle.dump(modelo_escolhido, buffer)  # 'modelo' é a sua variável do modelo treinado
            byte_im = buffer.getvalue()

            # 2. O download_button já é o botão que o usuário vai clicar
            st.download_button(
                label="📥 Exportar o modelo",
                data=byte_im,
                file_name="modelo_treinado.pkl",
                mime="application/octet-stream"
            )
else:
    # Se o treino não foi feito, mostramos um aviso amigável
    st.info("Clique no botão 'Treinar Modelos' acima para gerar os resultados.")
# ==================================================
# PASSO 3: Simulador Dinâmico
# ==================================================

# Início da Simulação
if st.session_state.get('mostrar_simulador', False) and st.session_state.get('treino_concluido', False):
    st.markdown("---")
    nome_modelo = st.session_state['modelo_selecionado_nome']
    st.subheader(f"🔮 Simulador de Predição ({nome_modelo})")

    coluna_target = st.session_state['target_selecionado']
    features = [col for col in df.columns if col != coluna_target]

    with st.form("form_predicao"):
        inputs_usuario = {}
        cols = st.columns(3)

        for i, col_nome in enumerate(features):
            with cols[i % 3]:
                # Identifica se o campo deve ser número ou seleção
                if pd.api.types.is_numeric_dtype(df[col_nome]):
                    media_val = float(df[col_nome].mean())
                    inputs_usuario[col_nome] = st.number_input(f"{col_nome}", value=media_val)
                else:
                    opcoes = df[col_nome].unique().tolist()
                    inputs_usuario[col_nome] = st.selectbox(f"{col_nome}", opcoes)

        botao_calcular = st.form_submit_button("✨ Calcular Predição")

    if botao_calcular:
        # 1. Preparar dados
        df_input = pd.DataFrame([inputs_usuario])
        # 2. Aplicar Scaler
        scaler = st.session_state['scaler_objeto']
        df_input_escalonado = scaler.transform(df_input)
        # 3. Predição
        modelo_final = st.session_state['modelos_objetos'][nome_modelo]
        resultado = modelo_final.predict(df_input_escalonado)
        # 4. Exibição Estilizada do Resultado
        st.markdown(f"""
                <div style="padding:20px; border-radius:10px; background-color:#1E3A8A; color:white; text-align:center; border: 2px solid #10B981;">
                    <h2 style="margin:0; font-family: sans-serif;">🎯 Resultado da Predição</h2>
                    <p style="font-size:18px; margin:10px 0;">Com base no modelo <b>{nome_modelo}</b>, o valor previsto para <b>{coluna_target}</b> é:</p>
                    <h1 style="margin:0; color:#10B981; font-size:48px;">{resultado[0]}</h1>
                </div>
                """, unsafe_allow_html=True)