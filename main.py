import streamlit as st
import io
# import pdfplumber # Removido
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError
from PIL import Image
import os
import google.generativeai as genai
# Import specific types for exception handling
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
MODEL_NAME = "gemini-2.5-pro-exp-03-25"
PAGES_PER_BATCH = 2

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(_pdf_bytes):
    """Converts PDF bytes into a list of PIL Image objects."""
    # ... (código da função convert_pdf_to_images sem alterações) ...
    images = []
    error_message = None
    try:
        thread_count = os.cpu_count() if os.cpu_count() else 2
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=thread_count)
    except PDFInfoNotInstalledError:
        error_message = "Erro de Configuração: Poppler não encontrado..." # Mensagem completa omitida por brevidade
    except PDFPageCountError:
        error_message = "Erro: Não foi possível determinar o número de páginas no PDF..."
    except PDFSyntaxError:
        error_message = "Erro: Sintaxe inválida no PDF..."
    except Exception as e:
        error_message = f"Erro inesperado durante a conversão: {str(e)}"
    if not images and not error_message:
         error_message = "Nenhuma imagem gerada..."
    return images, error_message


# --- Gemini Multimodal Analysis Function ---
def analyze_pages_with_gemini_multimodal(api_key, page_images_batch):
    """Analyzes a batch of PDF page images using Gemini."""
    # ... (código da função analyze_pages_with_gemini_multimodal sem alterações) ...
    analysis_output = f"## Análise das Páginas (Batch de {len(page_images_batch)})\n\n"
    full_analysis_text = ""
    if not page_images_batch: return "Nenhuma imagem de página fornecida."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        prompt_parts = [
            "**Instrução Principal:** ... (prompt completo omitido por brevidade) ...",
            "\n\n**IMAGENS DAS PÁGINAS PARA ANÁLISE:**\n"
        ]
        for img in page_images_batch:
            buffer = io.BytesIO()
            try:
                 img.save(buffer, format="PNG")
                 mime_type = "image/png"
            except Exception as e_save:
                 st.warning(f"Falha ao salvar imagem como PNG ({e_save}), pulando.")
                 continue
            image_bytes = buffer.getvalue()
            prompt_parts.append({"mime_type": mime_type, "data": image_bytes})

        with st.spinner(f"Analisando {len(page_images_batch)} página(s) com IA ({MODEL_NAME})..."):
            try:
                response = model.generate_content(prompt_parts, stream=False)
                # Process Response ... (código de processamento da resposta omitido)
                if hasattr(response, 'text'):
                     full_analysis_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                     full_analysis_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    # ... (tratamento de bloqueio) ...
                    full_analysis_text = f"**Análise Bloqueada pela API:** ..."
                    st.error(f"Análise bloqueada: ...")
                else:
                     full_analysis_text = "Resposta da API vazia ou inesperada."
                     st.warning(f"Resposta inesperada: {response}")

            except StopCandidateException as stop_e:
                 full_analysis_text = f"\n\n**Erro de Geração:** ... ({stop_e})."
                 st.error(f"Erro na Geração Gemini (StopCandidateException): {stop_e}")
            except Exception as e:
                 st.error(f"Erro durante a chamada da API Gemini: {str(e)}")
                 full_analysis_text += f"\n\n**Erro Crítico na Análise:** ... {str(e)}"
        analysis_output += full_analysis_text
    except Exception as e:
        st.error(f"Erro geral na análise: {str(e)}")
        analysis_output += f"\n\n**Erro Crítico:** {str(e)}"
    return analysis_output


# --- Callback Function ---
def sync_batch_selection():
    """Callback para selectbox, ajuda a sincronizar o estado."""
    # A ação principal é feita pelo 'key', este callback ajuda no fluxo.
    print(f"Callback sync_batch_selection: st.session_state.selected_batch is now '{st.session_state.get('selected_batch')}'")
    pass


# --- Streamlit Interface ---
st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"Envie um PDF...") # Mensagem completa omitida

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password")

    st.subheader("Opções de Análise")
    # Batch selection populated after upload

    st.markdown("---")
    st.markdown("### Como Usar: ...") # Instruções omitidas
    st.markdown("---")
    st.info("...")
    st.warning("**Dependência Externa:** Requer `poppler`...")

# --- Main Area Logic ---
# Initialize session state
default_state = {
    'analysis_result': None, 'error_message': None, 'pdf_page_images': [],
    'analysis_running': False, 'uploaded_file_id': None, 'batch_options': [],
    'selected_batch': None, 'total_pages': 0, 'original_filename': None
}
for key, value in default_state.items():
    if key not in st.session_state: st.session_state[key] = value

st.write("## 📄 1. Upload da Prova (PDF)")
uploaded_file = st.file_uploader("Selecione o arquivo PDF", type=["pdf"], key="file_uploader")

# --- Logic after file upload ---
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
    if current_file_id != st.session_state.uploaded_file_id:
        st.info(f"Novo arquivo: '{uploaded_file.name}'. Processando...")
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.original_filename = uploaded_file.name
        # Reset state
        for key in default_state:
            if key not in ['uploaded_file_id', 'original_filename']:
                st.session_state[key] = default_state[key]

        # --- Convert PDF ---
        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error:
            st.error(f"Falha na Conversão: {error}")
            st.session_state.error_message = f"Falha na Conversão: {error}"
            st.session_state.pdf_page_images = []
        elif not images:
            st.warning("Nenhuma imagem gerada.")
            st.session_state.error_message = "Nenhuma imagem gerada."
            st.session_state.pdf_page_images = []
        else:
            # --- Success: Store images, generate options, set initial selection ---
            st.session_state.pdf_page_images = images
            st.session_state.total_pages = len(images)
            st.success(f"Conversão OK! {st.session_state.total_pages} páginas prontas.")

            num_batches = math.ceil(st.session_state.total_pages / PAGES_PER_BATCH)
            batch_opts = []
            if st.session_state.total_pages > 1: batch_opts.append("Analisar Todas")
            for i in range(num_batches):
                start = i * PAGES_PER_BATCH + 1
                end = min((i + 1) * PAGES_PER_BATCH, st.session_state.total_pages)
                batch_opts.append(f"Página {start}" if start == end else f"Páginas {start}-{end}")

            st.session_state.batch_options = batch_opts
            # Set initial default (first specific batch or first option)
            if len(batch_opts) > 1 and "Analisar Todas" in batch_opts:
                 st.session_state.selected_batch = batch_opts[1]
            elif batch_opts:
                 st.session_state.selected_batch = batch_opts[0]
            else:
                 st.session_state.selected_batch = None
            st.rerun() # Update UI after conversion and option generation

# --- Display file details and batch selection UI (if images are ready) ---
if st.session_state.pdf_page_images:
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} pronto ({st.session_state.total_pages} págs).")

    with st.expander("Visualizar Miniaturas"):
        # ... (código do expander sem alterações) ...
        max_preview = 10; cols = st.columns(5)
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
            with cols[i % 5]: st.image(img, caption=f"Pág {i+1}", width=120)
        if st.session_state.total_pages > max_preview: st.markdown(f"*(Primeiras {max_preview})*")


    # --- Batch Selection UI (Sidebar) ---
    with st.sidebar:
         st.subheader("🎯 Selecionar Batch de Páginas")
         if st.session_state.batch_options:
              # Calcula o índice para exibição inicial (não crucial para a lógica de estado)
              current_selection_sidebar = st.session_state.get('selected_batch')
              try:
                   current_index_sidebar = st.session_state.batch_options.index(current_selection_sidebar)
              except (ValueError, IndexError):
                   current_index_sidebar = 0 # Default para primeiro item

              st.selectbox(
                  "Escolha o intervalo de páginas:",
                  options=st.session_state.batch_options,
                  index=current_index_sidebar,
                  key='selected_batch', # Vincula ao estado
                  on_change=sync_batch_selection, # Callback para ajudar na sincronia
                  help="Selecione as páginas para enviar à IA."
              )
         else:
              st.info("Faça upload de um PDF.")

    # --- Analysis Trigger ---
    st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")

    # ***** MODIFICAÇÃO CHAVE: Ler o estado AQUI *****
    current_selected_batch_for_button = st.session_state.get('selected_batch')
    button_label_batch = current_selected_batch_for_button if current_selected_batch_for_button else "None"

    # ***** DEBUGGING: Verifique o valor lido *****
    st.warning(f"DEBUG (Before Button): current_selected_batch_for_button = '{current_selected_batch_for_button}'")
    st.warning(f"DEBUG (Before Button): st.session_state.analysis_running = {st.session_state.analysis_running}")
    # ***** FIM DEBUGGING *****

    # Define se o botão deve estar desabilitado
    is_disabled = (
        st.session_state.analysis_running or
        not current_selected_batch_for_button or  # Verifica o valor lido AGORA
        not st.session_state.pdf_page_images or
        not api_key
    )

    analyze_button = st.button(
         f"Analisar Batch Selecionado ({button_label_batch})", # Usa o valor lido AGORA
         type="primary",
         use_container_width=True,
         disabled=is_disabled # Usa a variável de desabilitado calculada AGORA
    )

    if analyze_button:
        # Verificações pré-análise (usando o valor já verificado)
        if not api_key: st.error("⚠️ Insira a Chave API."); st.stop()
        if not current_selected_batch_for_button: st.error("⚠️ Selecione um batch."); st.stop() # Redundante devido ao disabled, mas seguro
        if not st.session_state.pdf_page_images: st.error("⚠️ Sem imagens PDF."); st.stop()

        # --- Iniciar processo ---
        st.session_state.analysis_running = True
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        # st.rerun() # REMOVIDO daqui - o fluxo continua para o bloco 'if analysis_running'

# --- Handle Analysis Execution ---
if st.session_state.analysis_running:
     # Usa o valor do estado que foi usado para iniciar
     batch_to_analyze = st.session_state.selected_batch
     with st.spinner(f"Preparando e analisando o batch '{batch_to_analyze}'..."):
        # --- Determina as páginas ---
        pages_to_process = []
        # ... (lógica para parsear 'batch_to_analyze' e obter 'pages_to_process' - sem alterações) ...
        all_images = st.session_state.pdf_page_images
        total_pg = st.session_state.total_pages
        if batch_to_analyze == "Analisar Todas":
            pages_to_process = all_images; st.info(f"Processando todas {total_pg} págs.")
        else:
            nums_str = re.findall(r'\d+', batch_to_analyze); # ... (resto do parsing) ...
            try: # Adicionado try/except robusto
                if len(nums_str) == 1: start_label, end_label = int(nums_str[0]), int(nums_str[0])
                elif len(nums_str) == 2: start_label, end_label = int(nums_str[0]), int(nums_str[1])
                else: raise ValueError("Formato batch inválido")
                start_idx, end_idx = start_label - 1, end_label
                if 0 <= start_idx < total_pg and start_idx < end_idx <= total_pg:
                    pages_to_process = all_images[start_idx:end_idx]
                    st.info(f"Analisando págs {start_label}-{end_label}...")
                else: st.warning(f"Intervalo inválido {start_label}-{end_label}.")
            except Exception as e: st.error(f"Erro no parsing do batch '{batch_to_analyze}': {e}")


        # --- Executa a análise ---
        if pages_to_process:
            analysis_result_text = analyze_pages_with_gemini_multimodal(api_key, pages_to_process)
            st.session_state.analysis_result = analysis_result_text
            if "Erro" in (analysis_result_text or "") or "Bloqueada" in (analysis_result_text or ""):
                 st.session_state.error_message = "Erro durante a análise pela IA."
        else:
            if not st.session_state.error_message:
                  st.session_state.error_message = "Nenhuma página selecionada/válida para análise neste batch."

        # --- Análise concluída ---
        st.session_state.analysis_running = False
        st.rerun() # Atualiza a UI para mostrar resultado/erro e reabilitar botão

# --- Display Results or Errors ---
if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}")

if st.session_state.analysis_result and not st.session_state.analysis_running:
    # ... (código para exibir resultado e botão de download - sem alterações) ...
    st.write(f"## 📊 3. Resultado (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False)
    try: # Download button
        # ... (lógica do nome do arquivo) ...
        original_fn = re.sub(r'[^\w\d-]+','_', os.path.splitext(st.session_state.original_filename or 'prova')[0])
        batch_sfx = re.sub(r'[^\w\d-]+','_', st.session_state.selected_batch or 'completo').strip('_')
        dl_fn = f"analise_multimodal_{original_fn}_batch_{batch_sfx}.md"
        st.download_button(f"📥 Baixar Análise ({batch_sfx})", (st.session_state.analysis_result or "").encode('utf-8'), dl_fn, "text/markdown")
    except Exception as dl_e: st.warning(f"Erro download: {dl_e}")


# --- Footer ---
st.markdown("---")
st.markdown(f"**Desenvolvido...** | Modelo: {MODEL_NAME} | **Requer Poppler**")