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
import base64 # Necessário para a função auxiliar removida

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
    images = []
    error_message = None
    try:
        thread_count = os.cpu_count() if os.cpu_count() else 2
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=thread_count) # PNG é mais seguro
    except PDFInfoNotInstalledError: error_message = "Erro Config: Poppler não encontrado..."
    except PDFPageCountError: error_message = "Erro: Não foi possível contar páginas PDF..."
    except PDFSyntaxError: error_message = "Erro: Sintaxe PDF inválida..."
    except Exception as e: error_message = f"Erro inesperado na conversão: {str(e)}"
    if not images and not error_message: error_message = "Nenhuma imagem gerada do PDF."
    return images, error_message

# --- Gemini Multimodal Analysis Function ---
def analyze_pages_with_gemini_multimodal(api_key, page_images_batch):
    """Analyzes a batch of PDF page images using Gemini."""
    # ... (código da função analyze_pages_with_gemini_multimodal, sem alterações significativas) ...
    analysis_output = f"## Análise (Batch: {len(page_images_batch)} págs)\n\n"
    full_analysis_text = ""
    if not page_images_batch: return "Nenhuma imagem fornecida para este batch."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=MODEL_NAME)
        prompt_parts = ["**Instrução Principal:** ... (prompt completo omitido)...", "\n\n**IMAGENS:**\n"]
        for img in page_images_batch:
            buffer = io.BytesIO()
            try:
                 img.save(buffer, format="PNG")
                 mime_type = "image/png"
            except Exception as e_save:
                 st.warning(f"Falha ao salvar img ({e_save}), pulando.")
                 continue
            image_bytes = buffer.getvalue()
            prompt_parts.append({"mime_type": mime_type, "data": image_bytes})

        with st.spinner(f"Analisando {len(page_images_batch)} pág(s) com IA ({MODEL_NAME})..."):
            try:
                response = model.generate_content(prompt_parts, stream=False)
                # Process Response ...
                if hasattr(response, 'text'): full_analysis_text = response.text
                elif hasattr(response, 'parts'): full_analysis_text = "".join(p.text for p in response.parts if hasattr(p,"text"))
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    full_analysis_text = f"**Análise Bloqueada:** {response.prompt_feedback.block_reason_message or response.prompt_feedback.block_reason}"
                    st.error(f"Análise bloqueada: {full_analysis_text}")
                else: full_analysis_text = "Resposta API vazia/inesperada."
            except StopCandidateException as e: full_analysis_text = f"**Erro Geração:** {e}"; st.error(f"{e}")
            except Exception as e: full_analysis_text = f"**Erro API:** {e}"; st.error(f"{e}")
        analysis_output += full_analysis_text
    except Exception as e: analysis_output += f"**Erro Crítico:** {e}"; st.error(f"{e}")
    return analysis_output

# --- Streamlit Interface ---
st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"Envie um PDF...")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password")
    st.subheader("Opções de Análise")

    # --- FORMULÁRIO PARA SELEÇÃO DE BATCH ---
    if st.session_state.get('batch_options'):
        with st.form("batch_selection_form"):
            st.markdown("**1. Escolha o Batch:**")

            options_sb = st.session_state.get('batch_options', [])
            # O índice default aqui deve refletir o valor ATUAL do selectbox (da key temporária)
            # ou o último valor confirmado (do estado principal), se a key temporária for None.
            # Isso garante que o selectbox não "salte" visualmente após a confirmação.
            val_temp = st.session_state.get('_temp_batch_selection_in_form')
            val_confirmado = st.session_state.get('selected_batch')
            val_para_indice = val_temp if val_temp is not None else val_confirmado

            try:
                idx_default = options_sb.index(val_para_indice) if val_para_indice in options_sb else 0
            except ValueError:
                 idx_default = 0

            # *** USA KEY TEMPORÁRIA ***
            st.selectbox(
                "Intervalo de páginas:",
                options=options_sb,
                index=idx_default,
                key='_temp_batch_selection_in_form', # Key temporária
                help="Selecione as páginas e clique em 'Confirmar'."
            )

            submitted = st.form_submit_button("🎯 Confirmar Batch Selecionado")

            if submitted:
                # *** ATUALIZA ESTADO PRINCIPAL EXPlicitamente ***
                st.session_state.selected_batch = st.session_state._temp_batch_selection_in_form
                # Opcional: Limpar o estado temporário após usar? Pode não ser necessário.
                # st.session_state._temp_batch_selection_in_form = None
                st.success(f"Batch '{st.session_state.selected_batch}' confirmado e pronto!") # Lê o estado principal atualizado
                print(f"Form Submitted: Main state 'selected_batch' updated to -> '{st.session_state.selected_batch}'") # Debug
    else:
        st.info("Faça upload de um PDF para habilitar a seleção.")
        if 'selected_batch' in st.session_state: st.session_state.selected_batch = None
        if '_temp_batch_selection_in_form' in st.session_state: st.session_state._temp_batch_selection_in_form = None


    st.markdown("---")
    st.markdown("### Como Usar:\n1. API Key.\n2. Upload.\n3. **Escolha + 'Confirmar'**.\n4. 'Analisar'.\n5. Repita.")
    st.markdown("---")
    st.info("...")
    st.warning("**Dependência Externa:** Requer `poppler`...")

# --- Main Area Logic ---
# Initialize session state (incluindo a key temporária)
default_state = {
    'analysis_result': None, 'error_message': None, 'pdf_page_images': [],
    'analysis_running': False, 'uploaded_file_id': None, 'batch_options': [],
    'selected_batch': None, # <<< Estado Principal
    '_temp_batch_selection_in_form': None, # <<< Estado Temporário do Widget
    'total_pages': 0, 'original_filename': None
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
        # Reset state (incluindo a key temporária)
        for key in default_state:
            if key not in ['uploaded_file_id', 'original_filename']: st.session_state[key] = default_state[key]

        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error or not images:
            err_msg = error if error else "Nenhuma imagem gerada."
            st.error(f"Falha: {err_msg}")
            st.session_state.error_message = f"Falha: {err_msg}"
            st.session_state.pdf_page_images = []
            st.session_state.batch_options = []
            st.session_state.selected_batch = None
            st.session_state._temp_batch_selection_in_form = None # Resetar também
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
            # Set initial selection (apenas se houver opções)
            initial_selection = None
            if batch_opts:
                initial_selection = batch_opts[1] if len(batch_opts) > 1 and "Analisar Todas" in batch_opts else batch_opts[0]
            st.session_state.selected_batch = initial_selection # Estado Principal
            st.session_state._temp_batch_selection_in_form = initial_selection # Estado Temporário

        st.rerun() # Update UI after processing upload

# --- Display file details and UI elements ---
if st.session_state.pdf_page_images:
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} pronto ({st.session_state.total_pages} págs).")

    with st.expander("Visualizar Miniaturas"):
        # ... (thumbnail display unchanged) ...
        max_preview = 10; cols = st.columns(5)
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
             with cols[i % 5]: st.image(img, caption=f"Pág {i+1}", width=120)
        if st.session_state.total_pages > max_preview: st.markdown(f"*(Primeiras {max_preview})*")


    # --- Analysis Trigger Button (Main Area) ---
    st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")

    # *** LÊ O ESTADO PRINCIPAL 'selected_batch' ***
    batch_confirmado_main = st.session_state.get('selected_batch')
    button_label_main = batch_confirmado_main if batch_confirmado_main else "None"

    st.info(f"Batch confirmado pronto para análise: **{button_label_main}**")
    # st.warning(f"DEBUG (Main - Before Button): batch_confirmado_main = '{batch_confirmado_main}'")

    is_disabled_main = (
        st.session_state.analysis_running or
        not batch_confirmado_main or # Verifica o estado principal
        not st.session_state.pdf_page_images or
        not api_key
    )

    analyze_button_main = st.button(
         f"Analisar Batch Confirmado ({button_label_main})",
         type="primary",
         use_container_width=True,
         disabled=is_disabled_main
    )

    if analyze_button_main:
        if not api_key: st.error("⚠️ Insira a Chave API."); st.stop()
        if not batch_confirmado_main: st.error("⚠️ Confirme um batch na sidebar."); st.stop()

        st.session_state.analysis_running = True
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        # Rerun implícito

# --- Handle Analysis Execution ---
if st.session_state.analysis_running:
     # *** USA O ESTADO PRINCIPAL 'selected_batch' ***
     batch_to_analyze_run = st.session_state.selected_batch
     with st.spinner(f"Preparando e analisando o batch '{batch_to_analyze_run}'..."):
        # --- Determina as páginas ---
        pages_to_process_run = []
        # ... (lógica para parsear 'batch_to_analyze_run' e obter 'pages_to_process_run' - sem alterações) ...
        all_images_run = st.session_state.pdf_page_images
        total_pg_run = st.session_state.total_pages
        if batch_to_analyze_run == "Analisar Todas": pages_to_process_run = all_images_run; st.info(f"Proc. todas {total_pg_run} págs.")
        else:
            nums_str_run = re.findall(r'\d+', str(batch_to_analyze_run)) # Add str()
            try:
                if len(nums_str_run)==1: s,e = int(nums_str_run[0]),int(nums_str_run[0])
                elif len(nums_str_run)==2: s,e = int(nums_str_run[0]),int(nums_str_run[1])
                else: raise ValueError("Inválido")
                s_idx, e_idx = s-1, e
                if 0<=s_idx<total_pg_run and s_idx<e_idx<=total_pg_run: pages_to_process_run = all_images_run[s_idx:e_idx]; st.info(f"Analisando {s}-{e}...")
                else: st.warning(f"Intervalo {s}-{e} inválido.")
            except Exception as ex: st.error(f"Erro parsing batch '{batch_to_analyze_run}': {ex}")


        # --- Executa a análise ---
        if pages_to_process_run:
            analysis_result_text_run = analyze_pages_with_gemini_multimodal(api_key, pages_to_process_run)
            st.session_state.analysis_result = analysis_result_text_run
            if "Erro" in (analysis_result_text_run or "") or "Bloqueada" in (analysis_result_text_run or ""): st.session_state.error_message = "Erro na análise IA."
        else:
            if not st.session_state.error_message: st.session_state.error_message = "Nenhuma pág. selecionada/válida."

        st.session_state.analysis_running = False
        st.rerun() # Atualiza a UI pós-análise

# --- Display Results or Errors ---
if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}")

if st.session_state.analysis_result and not st.session_state.analysis_running:
    # *** USA O ESTADO PRINCIPAL 'selected_batch' ***
    st.write(f"## 📊 3. Resultado (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False)
    try: # Download button
        # ... (lógica do nome do arquivo usando st.session_state.selected_batch) ...
        original_fn = re.sub(r'[^\w\d-]+','_', os.path.splitext(st.session_state.original_filename or 'prova')[0])
        batch_sfx = re.sub(r'[^\w\d-]+','_', st.session_state.selected_batch or 'completo').strip('_')
        dl_fn = f"analise_multimodal_{original_fn}_batch_{batch_sfx}.md"
        st.download_button(f"📥 Baixar Análise ({batch_sfx})", (st.session_state.analysis_result or "").encode('utf-8'), dl_fn, "text/markdown")
    except Exception as dl_e: st.warning(f"Erro download: {dl_e}")

# --- Footer ---
st.markdown("---")
st.markdown(f"**Desenvolvido...** | Modelo: {MODEL_NAME} | **Requer Poppler**")