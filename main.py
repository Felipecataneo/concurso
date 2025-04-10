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
MODEL_NAME = "gemini-1.5-pro-latest"
PAGES_PER_BATCH = 2

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(_pdf_bytes):
    """Converts PDF bytes into a list of PIL Image objects."""
    # ... (código da função convert_pdf_to_images) ...
    images = []
    error_message = None
    try:
        thread_count = os.cpu_count() if os.cpu_count() else 2
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=thread_count)
    except PDFInfoNotInstalledError: error_message = "Erro Config: Poppler não encontrado..."
    except PDFPageCountError: error_message = "Erro: Páginas PDF..."
    except PDFSyntaxError: error_message = "Erro: Sintaxe PDF..."
    except Exception as e: error_message = f"Erro conversão: {str(e)}"
    if not images and not error_message: error_message = "Nenhuma imagem gerada."
    return images, error_message

# --- Gemini Multimodal Analysis Function ---
def analyze_pages_with_gemini_multimodal(api_key, page_images_batch):
    """Analyzes a batch of PDF page images using Gemini."""
    # ... (código da função analyze_pages_with_gemini_multimodal) ...
    analysis_output = f"## Análise (Batch: {len(page_images_batch)} págs)\n\n"
    full_analysis_text = ""
    if not page_images_batch: return "Nenhuma imagem fornecida."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        prompt_parts = ["**Instrução Principal:** ...", "\n\n**IMAGENS:**\n"] # Prompt omitido
        for img in page_images_batch:
            buffer = io.BytesIO(); img.save(buffer, format="PNG"); image_bytes = buffer.getvalue()
            prompt_parts.append({"mime_type": "image/png", "data": image_bytes})

        with st.spinner(f"Analisando {len(page_images_batch)} pág(s) com IA..."):
            try:
                response = model.generate_content(prompt_parts, stream=False)
                # Process Response ... (código omitido)
                if hasattr(response, 'text'): full_analysis_text = response.text
                elif hasattr(response, 'parts'): full_analysis_text = "".join(p.text for p in response.parts if hasattr(p,"text"))
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    full_analysis_text = f"**Análise Bloqueada:** {response.prompt_feedback.block_reason_message}"
                    st.error(f"Análise bloqueada: {response.prompt_feedback.block_reason_message}")
                else: full_analysis_text = "Resposta API vazia/inesperada."
            except StopCandidateException as e: full_analysis_text = f"**Erro Geração:** {e}"; st.error(f"{e}")
            except Exception as e: full_analysis_text = f"**Erro API:** {e}"; st.error(f"{e}")
        analysis_output += full_analysis_text
    except Exception as e: analysis_output += f"**Erro Crítico:** {e}"; st.error(f"{e}")
    return analysis_output

# --- Callback Function ---
# REMOVIDO - Não mais necessário com a abordagem de formulário
# def sync_batch_selection():
#    pass

# --- Streamlit Interface ---
st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"Envie um PDF...")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password")

    st.subheader("Opções de Análise")

    # --- FORMULÁRIO PARA SELEÇÃO DE BATCH ---
    if st.session_state.get('batch_options'): # Verifica se há opções antes de criar o form
        with st.form("batch_selection_form"):
            st.markdown("**1. Escolha o Batch:**")

            current_selection_sidebar = st.session_state.get('selected_batch')
            try:
                # Apenas para definir a exibição inicial do selectbox
                current_index_sidebar = st.session_state.batch_options.index(current_selection_sidebar)
            except (ValueError, IndexError):
                current_index_sidebar = 0 # Default para o primeiro item

            # Selectbox DENTRO do formulário
            st.selectbox(
                "Intervalo de páginas:",
                options=st.session_state.batch_options,
                index=current_index_sidebar,
                key='selected_batch', # Key ainda útil para manter seleção entre runs não submetidos
                help="Selecione as páginas e clique em 'Confirmar'."
            )

            # Botão de submissão do formulário
            submitted = st.form_submit_button("🎯 Confirmar Batch Selecionado")

            if submitted:
                # Quando submetido, o valor de 'selected_batch' está atualizado
                st.success(f"Batch '{st.session_state.get('selected_batch')}' pronto para análise!")
                # A submissão do form já causa um rerun, não precisa de st.rerun() explícito
                print(f"Form Submitted: selected_batch = '{st.session_state.get('selected_batch')}'") # Debug no terminal
    else:
        st.info("Faça upload de um PDF para habilitar a seleção.")
        # Garantir que não há seleção se não há opções
        if 'selected_batch' in st.session_state:
            st.session_state.selected_batch = None

    st.markdown("---")
    st.markdown("### Como Usar:\n1. Cole a API Key.\n2. Upload PDF.\n3. **Selecione o batch e clique em 'Confirmar Batch'**.\n4. Clique em 'Analisar Batch Confirmado'.\n5. Repita 3 e 4.")
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
            if key not in ['uploaded_file_id', 'original_filename']: st.session_state[key] = default_state[key]

        # Convert PDF
        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error or not images:
            err_msg = error if error else "Nenhuma imagem gerada."
            st.error(f"Falha: {err_msg}")
            st.session_state.error_message = f"Falha: {err_msg}"
            st.session_state.pdf_page_images = []
            st.session_state.batch_options = [] # Limpa opções também
            st.session_state.selected_batch = None
        else:
            st.session_state.pdf_page_images = images
            st.session_state.total_pages = len(images)
            st.success(f"Conversão OK! {st.session_state.total_pages} págs.")
            # Generate options
            num_batches = math.ceil(st.session_state.total_pages / PAGES_PER_BATCH)
            batch_opts = []
            if st.session_state.total_pages > 1: batch_opts.append("Analisar Todas")
            for i in range(num_batches):
                s, e = i*PAGES_PER_BATCH+1, min((i+1)*PAGES_PER_BATCH, st.session_state.total_pages)
                batch_opts.append(f"Página {s}" if s == e else f"Páginas {s}-{e}")
            st.session_state.batch_options = batch_opts
            # Set initial selection (if options exist)
            if batch_opts:
                st.session_state.selected_batch = batch_opts[1] if len(batch_opts) > 1 and "Analisar Todas" in batch_opts else batch_opts[0]
            else:
                st.session_state.selected_batch = None

        st.rerun() # Rerun after processing upload


# --- Display file details and UI elements ---
if st.session_state.pdf_page_images:
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} pronto ({st.session_state.total_pages} págs).")

    with st.expander("Visualizar Miniaturas"):
        max_preview = 10; cols = st.columns(5)
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
            with cols[i % 5]: st.image(img, caption=f"Pág {i+1}", width=120)
        if st.session_state.total_pages > max_preview: st.markdown(f"*(Primeiras {max_preview})*")

    # --- Analysis Trigger Button (Main Area) ---
    st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")

    # Lê o valor ATUAL do batch selecionado (que foi confirmado pelo form)
    batch_confirmado = st.session_state.get('selected_batch')
    button_label = batch_confirmado if batch_confirmado else "None"

    # DEBUG: Mostra o valor que será usado pelo botão
    st.info(f"Batch confirmado pronto para análise: **{button_label}**")
    # st.warning(f"DEBUG (Main - Before Button): batch_confirmado = '{batch_confirmado}'")

    is_disabled = (
        st.session_state.analysis_running or
        not batch_confirmado or # Verifica se um batch foi confirmado
        not st.session_state.pdf_page_images or
        not api_key
    )

    analyze_button = st.button(
         f"Analisar Batch Confirmado ({button_label})", # Label reflete o batch confirmado
         type="primary",
         use_container_width=True,
         disabled=is_disabled
    )

    if analyze_button:
        if not api_key: st.error("⚠️ Insira a Chave API."); st.stop()
        if not batch_confirmado: st.error("⚠️ Confirme um batch na sidebar."); st.stop() # Redundante, mas seguro

        st.session_state.analysis_running = True
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        # Rerun IMPLÍCITO aqui, pois o estado mudou e o script re-executará

# --- Handle Analysis Execution ---
if st.session_state.analysis_running:
     # Usa o valor que foi confirmado e levou ao clique do botão
     batch_to_analyze = st.session_state.selected_batch
     with st.spinner(f"Preparando e analisando o batch '{batch_to_analyze}'..."):
        # --- Determina as páginas ---
        pages_to_process = []
        all_images = st.session_state.pdf_page_images
        total_pg = st.session_state.total_pages
        # ... (lógica para parsear 'batch_to_analyze' e obter 'pages_to_process') ...
        if batch_to_analyze == "Analisar Todas": pages_to_process = all_images; st.info(f"Proc. todas {total_pg} págs.")
        else:
            nums_str = re.findall(r'\d+', batch_to_analyze)
            try:
                if len(nums_str)==1: s,e = int(nums_str[0]),int(nums_str[0])
                elif len(nums_str)==2: s,e = int(nums_str[0]),int(nums_str[1])
                else: raise ValueError("Inválido")
                s_idx, e_idx = s-1, e
                if 0<=s_idx<total_pg and s_idx<e_idx<=total_pg: pages_to_process = all_images[s_idx:e_idx]; st.info(f"Analisando {s}-{e}...")
                else: st.warning(f"Intervalo {s}-{e} inválido.")
            except Exception as ex: st.error(f"Erro parsing batch '{batch_to_analyze}': {ex}")

        # --- Executa a análise ---
        if pages_to_process:
            analysis_result_text = analyze_pages_with_gemini_multimodal(api_key, pages_to_process)
            st.session_state.analysis_result = analysis_result_text
            if "Erro" in (analysis_result_text or "") or "Bloqueada" in (analysis_result_text or ""): st.session_state.error_message = "Erro na análise IA."
        else:
            if not st.session_state.error_message: st.session_state.error_message = "Nenhuma pág. selecionada/válida."

        # --- Análise concluída ---
        st.session_state.analysis_running = False
        st.rerun() # Atualiza a UI pós-análise

# --- Display Results or Errors ---
if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}")

if st.session_state.analysis_result and not st.session_state.analysis_running:
    # ... (código para exibir resultado e botão de download) ...
    st.write(f"## 📊 3. Resultado (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False)
    try: # Download button
        original_fn = re.sub(r'[^\w\d-]+','_', os.path.splitext(st.session_state.original_filename or 'prova')[0])
        batch_sfx = re.sub(r'[^\w\d-]+','_', st.session_state.selected_batch or 'completo').strip('_')
        dl_fn = f"analise_multimodal_{original_fn}_batch_{batch_sfx}.md"
        st.download_button(f"📥 Baixar Análise ({batch_sfx})", (st.session_state.analysis_result or "").encode('utf-8'), dl_fn, "text/markdown")
    except Exception as dl_e: st.warning(f"Erro download: {dl_e}")

# --- Footer ---
st.markdown("---")
st.markdown(f"**Desenvolvido...** | Modelo: {MODEL_NAME} | **Requer Poppler**")