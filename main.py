import streamlit as st
import io
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image
import os
import google.generativeai as genai
from google.generativeai.types import StopCandidateException
import re
import time
import math
import base64

# --- Page Configuration ---
st.set_page_config(
    page_title="Analisador Multimodal de Provas IA",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constantes ---
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Modelo mais recente e geralmente mais rápido/barato
PAGES_PER_BATCH = 2 # Analisar 2 páginas por vez

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(_pdf_bytes):
    """Converts PDF bytes into a list of PIL Image objects."""
    images = []
    error_message = None
    # st.info("Iniciando conversão de PDF para imagens...") # Removido log de depuração
    try:
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=os.cpu_count())
        if images: # Só mostra sucesso se realmente gerou imagens
             st.success(f"Conversão concluída: {len(images)} páginas geradas.") # Mantido feedback essencial
    except PDFInfoNotInstalledError:
        error_message = """
        Erro de Configuração: Poppler não encontrado.
        'pdf2image' requer a instalação do utilitário 'poppler'. Verifique as instruções de instalação para seu sistema.
        """
        st.error(error_message) # Mantido feedback essencial
    except PDFPageCountError:
        error_message = "Erro: Não foi possível determinar o número de páginas no PDF. O arquivo pode estar corrompido."
        st.error(error_message) # Mantido feedback essencial
    except PDFSyntaxError:
        error_message = "Erro: Sintaxe inválida no PDF. O arquivo pode estar corrompido ou mal formatado."
        st.error(error_message) # Mantido feedback essencial
    except Exception as e:
        error_message = f"Erro inesperado durante a conversão de PDF para imagem: {str(e)}"
        st.error(error_message) # Mantido feedback essencial

    if not images and not error_message:
         error_message = "Nenhuma imagem pôde ser gerada a partir do PDF. Verifique se o arquivo não está vazio ou protegido."
         st.warning(error_message) # Mantido feedback essencial

    return images, error_message

def analyze_pages_with_gemini_multimodal(api_key, page_images_batch):
    """
    Analyzes a batch of PDF page images using Gemini's multimodal capabilities.
    """
    analysis_output = f"## Análise das Páginas (Batch de {len(page_images_batch)})\n\n"
    full_analysis_text = ""

    if not page_images_batch:
        st.warning("Nenhuma imagem de página recebida para análise neste batch.") # Mantido feedback essencial
        return "Nenhuma imagem de página fornecida para este batch."

    # --- LOG Adicional Removido ---
    # st.info(f"[analyze_pages_with_gemini_multimodal] Recebeu {len(page_images_batch)} imagens para processar.")

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=MODEL_NAME)

        prompt_parts = [
            # ... (seu prompt extenso aqui - mantenha como está) ...
            "**Instrução Principal:** Você é um professor especialista analisando páginas de uma prova de concurso fornecidas como imagens. Sua tarefa é identificar TODAS as questões (com seus números, texto completo, alternativas A,B,C,D,E ou formato Certo/Errado) e qualquer texto de contexto associado (como 'Texto I') visíveis nas imagens a seguir.",
            "\n\n**Para CADA questão identificada nas imagens fornecidas, forneça uma análise DETALHADA e DIDÁTICA em formato Markdown, seguindo esta estrutura:**",
            "\n\n```markdown",
            "## Questão [Número da Questão] - Análise Detalhada",
            "",
            "### 1. Contexto Aplicado (se houver)",
            "*   Se a questão se refere a um texto base ('Texto I', 'Leia o texto...', etc.) visível nas imagens, resuma o ponto principal do contexto aqui.",
            "*   Se não houver contexto explícito, indique 'Nenhum contexto específico identificado para esta questão.'",
            "",
            "### 2. Transcrição da Questão/Item",
            "*   Transcreva o comando principal da questão e suas alternativas (A,B,C,D,E) ou a afirmação (Certo/Errado) EXATAMENTE como visto na imagem.",
            "",
            "### 3. Julgamento/Resposta Correta",
            "*   Indique **CERTO**/**ERRADO** ou a **Alternativa Correta** (ex: **Alternativa C**). Forneça apenas a resposta final aqui.",
            "",
            "### 4. Justificativa Completa",
            "*   Explique detalhadamente o raciocínio. **CRUCIAL:** Se houver contexto, explique COMO ele leva à resposta.",
            "*   Se C/E 'Errado', explique o erro. Se MC, explique por que a correta está certa E por que as outras alternativas estão erradas.",
            "",
            "### 5. Conhecimentos Avaliados",
            "*   Disciplina Principal e Assunto Específico.",
            "",
            "### 6. Dicas e Pegadinhas (Opcional)",
            "*   Há alguma dica útil ou pegadinha comum relacionada a esta questão?",
            "```",
            "\n\n**IMPORTANTE:** Analise TODAS as questões visíveis nas imagens a seguir. Se uma questão parecer continuar na próxima página (não incluída neste batch), mencione isso claramente na análise da questão. Apresente as análises das questões na ordem em que aparecem nas páginas.",
            "\n\n**IMAGENS DAS PÁGINAS PARA ANÁLISE:**\n"
        ]

        image_preparation_success = True
        prepared_image_parts = []

        for i, img in enumerate(page_images_batch):
            image_bytes = None
            mime_type = None
            with io.BytesIO() as buffer:
                # --- Logs de processamento de imagem removidos ---
                # st.info(f"Processando imagem {i+1}/{len(page_images_batch)} do batch...")
                try:
                    img.save(buffer, format="WEBP", lossless=True, quality=90)
                    mime_type = "image/webp"
                    image_bytes = buffer.getvalue()
                    # st.info(f"  Imagem {i+1} salva como WEBP ({len(image_bytes)} bytes).") # Removido
                except Exception as e_webp:
                    # Mantido o warning pois informa sobre um fallback importante
                    st.warning(f"Falha ao salvar imagem {i+1} como WEBP ({e_webp}), tentando PNG.")
                    buffer.seek(0)
                    buffer.truncate()
                    try:
                        img.save(buffer, format="PNG")
                        mime_type = "image/png"
                        image_bytes = buffer.getvalue()
                        # st.info(f"  Imagem {i+1} salva como PNG ({len(image_bytes)} bytes).") # Removido
                    except Exception as e_png:
                        # Mantido erro crítico
                        st.error(f"ERRO CRÍTICO: Falha ao salvar imagem {i+1} como PNG também: {e_png}")
                        image_bytes = None
                        image_preparation_success = False
                        break

            if image_bytes and mime_type:
                 prepared_image_parts.append({"mime_type": mime_type, "data": image_bytes})
            elif not image_preparation_success:
                 break

        if not image_preparation_success:
             # Mantido erro crítico
             st.error("Erro na preparação de uma ou mais imagens. Análise cancelada.")
             analysis_output += "\n\n**Erro Crítico:** Falha ao preparar imagens para análise."
             return analysis_output

        if not prepared_image_parts:
            # Mantido erro crítico
            st.error("Nenhuma imagem pôde ser preparada para este batch. Verifique as imagens de entrada ou a seleção.")
            analysis_output += "\n\n**Erro Crítico:** Nenhuma imagem válida para enviar à API neste batch."
            return analysis_output

        prompt_parts.extend(prepared_image_parts)

        # --- Log de envio removido ---
        # st.info(f"Enviando prompt com {len(prepared_image_parts)} imagens para a API Gemini ({MODEL_NAME})...")
        with st.spinner(f"Analisando {len(page_images_batch)} página(s) com IA Multimodal ({MODEL_NAME})... Esta etapa pode levar alguns minutos."):
            try:
                response = model.generate_content(prompt_parts, stream=False)

                # --- Log da resposta crua (comentado/removido) ---
                # st.json({"api_response": str(response)})

                if hasattr(response, 'text') and response.text:
                    full_analysis_text = response.text
                    # st.info("Texto da análise extraído diretamente do atributo 'text' da resposta.") # Removido
                elif hasattr(response, 'parts') and response.parts:
                     full_analysis_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                     # st.info("Texto da análise extraído concatenando 'parts' da resposta.") # Removido
                elif hasattr(response, 'prompt_feedback') and response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    block_message = getattr(response.prompt_feedback, 'block_reason_message', f"Reason code: {block_reason}")
                    full_analysis_text = f"**Análise Bloqueada pela API:** {block_message}"
                    # Mantido erro essencial
                    st.error(f"A análise multimodal foi bloqueada pela API: {block_message}")
                else:
                     full_analysis_text = "A API retornou uma resposta vazia ou em formato não esperado."
                     # Mantido warning essencial
                     st.warning(f"Resposta inesperada ou vazia da análise. Resposta recebida: {response}")

                # Mantido warning essencial
                if not full_analysis_text.strip() and "Bloqueada" not in full_analysis_text:
                    st.warning("O processamento da resposta da API resultou em texto vazio.")

            except StopCandidateException as stop_e:
                 full_analysis_text = f"\n\n**Erro de Geração:** A análise foi interrompida prematuramente. Causa provável: {stop_e}. Verifique as políticas de conteúdo ou tente novamente."
                 st.error(f"Erro na Geração Gemini (StopCandidateException): A resposta foi interrompida. Detalhes: {stop_e}") # Mantido
            except Exception as e:
                 st.error(f"Erro durante a chamada da API Gemini: {str(e)}", icon="🚨") # Mantido
                 full_analysis_text += f"\n\n**Erro Crítico na Análise:** Não foi possível completar a análise devido a um erro inesperado na API: {str(e)}"

        analysis_output += full_analysis_text

    except Exception as e:
        st.error(f"Erro geral durante a preparação ou análise multimodal: {str(e)}", icon="🔥") # Mantido
        analysis_output += f"\n\n**Erro Crítico:** Falha inesperada no setup da análise: {str(e)}"

    # --- Logs finais da função removidos ---
    # if not full_analysis_text.strip() and "Erro" not in analysis_output and "Bloqueada" not in analysis_output:
    #     st.warning(f"[analyze_pages_with_gemini_multimodal] Retornando análise aparentemente vazia para o batch.")
    # elif "Erro" in analysis_output or "Bloqueada" in analysis_output:
    #     st.error(f"[analyze_pages_with_gemini_multimodal] Retornando análise com erro/bloqueio para o batch.")
    # else:
    #     st.success(f"[analyze_pages_with_gemini_multimodal] Retornando análise bem-sucedida para o batch.")

    return analysis_output

# --- Streamlit Interface ---

st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"""
Envie um arquivo de prova em **PDF**. A ferramenta converterá as páginas em imagens e usará IA multimodal ({MODEL_NAME}) para identificar e analisar as questões **diretamente das imagens**.
Ideal para PDFs escaneados ou onde a extração de texto falha.
**Aviso:** Requer `poppler` instalado. O processamento pode levar alguns minutos por batch.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password", help=f"Necessária para usar o {MODEL_NAME}.")

    st.subheader("Opções de Análise")

    st.markdown("---")
    st.markdown(f"""
    ### Como Usar:
    1.  Cole sua chave API do Google Gemini.
    2.  Faça o upload do arquivo PDF.
    3.  Aguarde a conversão (pode levar um tempo).
    4.  Selecione o **batch de páginas** desejado na barra lateral.
    5.  Clique em "Analisar Batch Selecionado".
    6.  Aguarde a análise multimodal pela IA.
    7.  **Repita os passos 4-6 para outros batches do mesmo PDF.**
    8.  Visualize ou baixe o resultado do batch atual na área principal.
    """)
    st.markdown("---")
    st.info("A precisão depende da qualidade da imagem e da capacidade da IA. Verifique os resultados.") # Mantido
    st.warning("**Dependência Externa:** Requer `poppler` instalado no ambiente de execução.") # Mantido

# --- Main Area Logic ---

default_state = {
    'analysis_result': None,
    'error_message': None,
    'pdf_page_images': [],
    'analysis_running': False,
    'uploaded_file_id': None,
    'batch_options': [],
    'selected_batch': None,
    'total_pages': 0,
    'original_filename': None,
    'results_by_batch': {}
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.write("## 📄 1. Upload da Prova (PDF)")
uploaded_file = st.file_uploader(
    "Selecione o arquivo PDF",
    type=["pdf"],
    key="file_uploader_pdf_multimodal"
)

if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if current_file_id != st.session_state.uploaded_file_id:
        st.info(f"Novo arquivo detectado: '{uploaded_file.name}'. Iniciando processamento...") # Mantido
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.original_filename = uploaded_file.name
        # Reset state...
        st.session_state.pdf_page_images = []
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        st.session_state.batch_options = []
        st.session_state.selected_batch = None
        st.session_state.analysis_running = False
        st.session_state.results_by_batch = {}

        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error:
            st.session_state.error_message = f"Falha na Conversão do PDF: {error}"
            st.session_state.pdf_page_images = []
        elif not images:
            st.session_state.error_message = "Nenhuma imagem foi gerada a partir do PDF."
            st.session_state.pdf_page_images = []
        else:
            st.session_state.pdf_page_images = images
            st.session_state.total_pages = len(images)

            num_batches = math.ceil(st.session_state.total_pages / PAGES_PER_BATCH)
            batch_opts = []
            for i in range(num_batches):
                start_page = i * PAGES_PER_BATCH + 1
                end_page = min((i + 1) * PAGES_PER_BATCH, st.session_state.total_pages)
                if start_page == end_page:
                     batch_opts.append(f"Página {start_page}")
                else:
                     batch_opts.append(f"Páginas {start_page}-{end_page}")

            if num_batches > 1 and st.session_state.total_pages > 1:
                 batch_opts.append("Analisar Todas")

            st.session_state.batch_options = batch_opts
            if batch_opts:
                 st.session_state.selected_batch = batch_opts[0]
            else:
                 st.session_state.selected_batch = None

            # st.info("Opções de batch geradas. Selecione na barra lateral.") # Removido
            st.rerun()

if st.session_state.pdf_page_images:
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} processado. {st.session_state.total_pages} páginas prontas.") # Mantido

    with st.expander("Visualizar Páginas Convertidas (Miniaturas)"):
        max_preview = 10
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
            with cols[i % 5]:
                try:
                    st.image(img, caption=f"Página {i+1}", width=120)
                except Exception as img_disp_err:
                    # Mantido warning essencial
                    st.warning(f"Erro exibindo Pág {i+1}: {img_disp_err}")

        if st.session_state.total_pages > max_preview:
            st.markdown(f"*(Pré-visualização limitada às primeiras {max_preview} de {st.session_state.total_pages} páginas)*")

with st.sidebar:
    st.subheader("🎯 Selecionar Batch de Páginas")
    if st.session_state.batch_options:

        def update_batch_selection_callback():
            selected_value_from_widget = st.session_state.batch_selector_widget
            st.session_state.selected_batch = selected_value_from_widget
            # st.info(f"Callback: Batch selecionado alterado para '{st.session_state.selected_batch}'") # Removido

            if selected_value_from_widget in st.session_state.results_by_batch:
                st.session_state.analysis_result = st.session_state.results_by_batch[selected_value_from_widget]
                st.session_state.error_message = None
                st.sidebar.success(f"Carregado resultado existente para '{selected_value_from_widget}'") # Mantido
            else:
                 st.session_state.analysis_result = None
                 st.session_state.error_message = None
                 # st.sidebar.info(f"Batch '{selected_value_from_widget}' não analisado previamente.") # Removido

        try:
            if st.session_state.selected_batch not in st.session_state.batch_options:
                 if st.session_state.batch_options:
                      st.session_state.selected_batch = st.session_state.batch_options[0]
                 else:
                      st.session_state.selected_batch = None
            current_index = st.session_state.batch_options.index(st.session_state.selected_batch) if st.session_state.selected_batch else 0
        except (ValueError, TypeError):
            current_index = 0
            if st.session_state.batch_options:
                 st.session_state.selected_batch = st.session_state.batch_options[current_index]

        st.selectbox(
            "Escolha o intervalo de páginas:",
            options=st.session_state.batch_options,
            index=current_index,
            key="batch_selector_widget",
            on_change=update_batch_selection_callback,
            help="Selecione as páginas a serem enviadas para análise pela IA."
        )

        # --- Log de estado removido ---
        # st.sidebar.caption(f"Batch selecionado no estado: {st.session_state.selected_batch}")

    else:
        st.info("Faça upload de um PDF para ver as opções de batch.") # Mantido

    if st.session_state.results_by_batch:
        st.sidebar.subheader("📊 Batch(es) Analisado(s)")
        sorted_batches = sorted(st.session_state.results_by_batch.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf'))
        for batch_name in sorted_batches:
             # Mantidos indicadores visuais
             if "Erro Crítico" not in st.session_state.results_by_batch[batch_name] and \
                "Análise Bloqueada" not in st.session_state.results_by_batch[batch_name]:
                 st.sidebar.success(f"✅ {batch_name}")
             else:
                 st.sidebar.warning(f"⚠️ {batch_name} (com erro/bloqueio)")

    st.write("## ⚙️ 2. Iniciar Análise")
    selected_batch_display = st.session_state.get('selected_batch', 'Nenhum')
    batch_already_analyzed = selected_batch_display in st.session_state.results_by_batch and \
                           ("Erro Crítico" not in st.session_state.results_by_batch.get(selected_batch_display, "") and \
                            "Análise Bloqueada" not in st.session_state.results_by_batch.get(selected_batch_display, ""))


    button_text = f"Analisar Batch ({selected_batch_display})"
    if batch_already_analyzed:
        button_text = f"Reanalisar Batch ({selected_batch_display})"

    analyze_button = st.button(
         button_text,
         type="primary",
         use_container_width=True,
         disabled=st.session_state.analysis_running or not st.session_state.selected_batch or not st.session_state.pdf_page_images or not api_key
    )

    if analyze_button:
        # Mantidos erros essenciais
        if not api_key:
            st.error("⚠️ Por favor, insira sua Chave API do Google Gemini na barra lateral.")
        elif not st.session_state.selected_batch:
             st.error("⚠️ Por favor, selecione um batch de páginas na barra lateral.")
        elif not st.session_state.pdf_page_images:
             st.error("⚠️ Nenhuma imagem de página encontrada. Faça upload e converta um PDF primeiro.")
        else:
            # --- Log de início de análise removido ---
            # st.info(f"Iniciando análise para o batch: '{st.session_state.selected_batch}'...")
            st.session_state.analysis_running = True
            st.session_state.analysis_result = None
            st.session_state.error_message = None
            if st.session_state.selected_batch in st.session_state.results_by_batch:
                 del st.session_state.results_by_batch[st.session_state.selected_batch]
                 # st.info(f"Resultado anterior para '{st.session_state.selected_batch}' removido para reanálise.") # Removido

            st.rerun()

if st.session_state.analysis_running:
     with st.spinner(f"Preparando e analisando o batch '{st.session_state.selected_batch}'... Isso pode levar um tempo."):
        pages_to_process = []
        selected = st.session_state.selected_batch
        all_images = st.session_state.pdf_page_images
        total_pg = st.session_state.total_pages

        # --- LOGS DETALHADOS DA SELEÇÃO REMOVIDOS ---
        # st.info(f"Processando seleção de batch: '{selected}'")
        # st.info(f"Total de páginas disponíveis: {len(all_images)}")

        if selected == "Analisar Todas":
            pages_to_process = all_images
            # st.success(f"Selecionadas todas as {len(pages_to_process)} páginas para processamento.")
        elif selected:
            nums_str = re.findall(r'\d+', selected)
            try:
                if len(nums_str) == 1:
                    start_page_label = int(nums_str[0])
                    end_page_label = start_page_label
                elif len(nums_str) == 2:
                    start_page_label = int(nums_str[0])
                    end_page_label = int(nums_str[1])
                else:
                    raise ValueError(f"Formato de batch inesperado: {selected}")

                start_index = start_page_label - 1
                end_index = end_page_label

                # --- LOG DE ÍNDICES REMOVIDO ---
                # st.info(f"Batch: '{selected}' -> Páginas (label): {start_page_label}-{end_page_label}")
                # st.info(f"Convertido para Índices (0-based): start_index={start_index}, end_index={end_index} (para slice)")

                if 0 <= start_index < total_pg and start_index < end_index <= total_pg:
                    pages_to_process = all_images[start_index:end_index]
                    # st.success(f"Slice bem-sucedido. {len(pages_to_process)} páginas selecionadas (índices {start_index} a {end_index-1}) para o batch '{selected}'.")
                else:
                    # Mantido erro essencial
                    st.error(f"Erro de Índice: Intervalo de páginas inválido (labels {start_page_label}-{end_page_label} / índices {start_index}-{end_index}) para o total de {total_pg} páginas. Batch: '{selected}'.")
                    pages_to_process = []

            except (ValueError, IndexError) as parse_e:
                 # Mantido erro essencial
                st.error(f"Erro ao interpretar ou fatiar a seleção de batch '{selected}': {parse_e}")
                pages_to_process = []
        else:
             # Mantido erro essencial
             st.error("Nenhum batch válido selecionado para análise.")
             pages_to_process = []

        analysis_markdown = None
        if pages_to_process:
            # st.info(f"Enviando {len(pages_to_process)} imagens para a função de análise multimodal...") # Removido
            analysis_markdown = analyze_pages_with_gemini_multimodal(
                    api_key,
                    pages_to_process,
                )

            st.session_state.analysis_result = analysis_markdown

            if analysis_markdown and "Erro Crítico" not in analysis_markdown and "Análise Bloqueada" not in analysis_markdown:
                 st.session_state.results_by_batch[selected] = analysis_markdown
                 # st.success(f"Análise para o batch '{selected}' concluída e armazenada.") # Removido (implícito pela exibição)
            else:
                 st.session_state.error_message = f"A análise do batch '{selected}' retornou um erro ou foi bloqueada. Veja detalhes abaixo."
                 # Mantido erro que define o estado de erro
                 st.error(f"Análise para '{selected}' falhou ou foi bloqueada. Resultado não armazenado como sucesso.")
                 if selected in st.session_state.results_by_batch:
                      del st.session_state.results_by_batch[selected]

        else:
            # Mantido erro essencial
            st.error(f"Nenhuma página foi selecionada para análise no batch '{selected}' devido a erro anterior.")
            st.session_state.error_message = f"Falha ao selecionar páginas para o batch '{selected}'. Verifique os logs acima."
            st.session_state.analysis_result = None

        st.session_state.analysis_running = False
        st.rerun()

# --- Exibir Resultados ou Erros ---

if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}") # Mantido
    if st.session_state.analysis_result and ("Erro Crítico" in st.session_state.analysis_result or "Análise Bloqueada" in st.session_state.analysis_result) :
         st.warning("Detalhes do erro/resposta da API:") # Mantido
         st.markdown(st.session_state.analysis_result)

elif st.session_state.analysis_result and not st.session_state.analysis_running:
    st.write(f"## 📊 3. Resultado da Análise Multimodal (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False)

    try:
        original_filename_base = "prova"
        if st.session_state.original_filename:
             original_filename_base = os.path.splitext(st.session_state.original_filename)[0]
             original_filename_base = re.sub(r'[^\w\d-]+', '_', original_filename_base)

        batch_suffix = "completo"
        if st.session_state.selected_batch and st.session_state.selected_batch != "Analisar Todas":
             nums = re.findall(r'\d+', st.session_state.selected_batch)
             if len(nums) == 1: batch_suffix = f"pag_{nums[0]}"
             elif len(nums) == 2: batch_suffix = f"pags_{nums[0]}-{nums[1]}"
             else: batch_suffix = re.sub(r'[^\w\d-]+', '_', st.session_state.selected_batch).strip('_')
        elif st.session_state.selected_batch == "Analisar Todas":
             batch_suffix = "todas"

        download_filename = f"analise_multimodal_{original_filename_base}_batch_{batch_suffix}.md"

        st.download_button(
            label=f"📥 Baixar Análise do Batch Atual ({st.session_state.get('selected_batch', 'N/A')}) (Markdown)",
            data=st.session_state.analysis_result.encode('utf-8'),
            file_name=download_filename,
            mime="text/markdown"
        )

    except Exception as dl_e:
        st.warning(f"Não foi possível gerar o botão de download para o batch atual: {dl_e}") # Mantido

if len(st.session_state.results_by_batch) > 1 and not st.session_state.analysis_running:
     st.write("---")
     st.write("### Download Combinado")
     try:
          all_analyses = []
          sorted_batches = sorted(st.session_state.results_by_batch.keys(), key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else float('inf'))

          for batch_name in sorted_batches:
               header = f"# Análise do Batch: {batch_name}\n\n"
               all_analyses.append(header + st.session_state.results_by_batch[batch_name])

          combined_analysis_text = "\n\n---\n\n".join(all_analyses)

          original_filename_base = "prova"
          if st.session_state.original_filename:
               original_filename_base = os.path.splitext(st.session_state.original_filename)[0]
               original_filename_base = re.sub(r'[^\w\d-]+', '_', original_filename_base)

          combined_filename = f"analise_multimodal_{original_filename_base}_COMPLETA_{len(st.session_state.results_by_batch)}_batches.md"

          st.download_button(
                label=f"📥 Baixar TODAS as Análises Combinadas ({len(st.session_state.results_by_batch)} batches) (Markdown)",
                data=combined_analysis_text.encode('utf-8'),
                file_name=combined_filename,
                mime="text/markdown"
            )
     except Exception as dl_all_e:
          st.warning(f"Não foi possível gerar o botão de download combinado: {dl_all_e}") # Mantido

if not st.session_state.uploaded_file_id:
     st.info("Aguardando upload do arquivo PDF...") # Mantido