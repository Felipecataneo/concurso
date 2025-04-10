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
MODEL_NAME = "gemini-2.5-pro-exp-03-25" # Ou "gemini-pro-vision" se preferir
PAGES_PER_BATCH = 2 # Analisar 2 páginas por vez

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(_pdf_bytes): # Renomeado argumento para evitar shadowing
    """Converts PDF bytes into a list of PIL Image objects."""
    images = []
    error_message = None
    try:
        # Tenta usar múltiplos threads para acelerar, se possível
        thread_count = os.cpu_count() if os.cpu_count() else 2
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=thread_count)
    except PDFInfoNotInstalledError:
        error_message = """
        Erro de Configuração: Poppler não encontrado.
        'pdf2image' requer a instalação do utilitário 'poppler'. Verifique as instruções de instalação para seu sistema (Linux: sudo apt-get install poppler-utils, macOS: brew install poppler, Windows: download e add ao PATH).
        """
    except PDFPageCountError:
        error_message = "Erro: Não foi possível determinar o número de páginas no PDF. O arquivo pode estar corrompido."
    except PDFSyntaxError:
        error_message = "Erro: Sintaxe inválida no PDF. O arquivo pode estar corrompido ou mal formatado."
    except Exception as e:
        error_message = f"Erro inesperado durante a conversão de PDF para imagem: {str(e)}"

    if not images and not error_message:
         error_message = "Nenhuma imagem pôde ser gerada a partir do PDF. Verifique se o arquivo não está vazio ou protegido."

    return images, error_message

# --- Gemini Multimodal Analysis Function ---
def analyze_pages_with_gemini_multimodal(api_key, page_images_batch):
    """
    Analyzes a batch of PDF page images using Gemini's multimodal capabilities.
    """
    analysis_output = f"## Análise das Páginas (Batch de {len(page_images_batch)})\n\n"
    full_analysis_text = ""

    if not page_images_batch:
        st.warning("Nenhuma imagem de página recebida para análise neste batch.")
        return "Nenhuma imagem de página fornecida para este batch."

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            # safety_settings={ # Exemplo: Ajustar segurança se necessário
            #     'HATE': 'BLOCK_NONE',
            #     'HARASSMENT': 'BLOCK_NONE',
            #     'SEXUAL' : 'BLOCK_NONE',
            #     'DANGEROUS' : 'BLOCK_NONE'
            # }
        )

        # --- Construct the Multimodal Prompt ---
        prompt_parts = [
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

        # Add images to the prompt
        for img in page_images_batch:
            buffer = io.BytesIO()
            try:
                 # WEBP lossless é geralmente bom, mas pode aumentar o tamanho; PNG é seguro.
                 # img.save(buffer, format="WEBP", lossless=True)
                 # mime_type = "image/webp"
                 img.save(buffer, format="PNG") # PNG é mais compatível
                 mime_type = "image/png"
            except Exception as e_save:
                 st.warning(f"Falha ao salvar imagem como PNG ({e_save}), pulando esta imagem.")
                 continue # Pula para a próxima imagem se houver erro

            image_bytes = buffer.getvalue()
            prompt_parts.append({"mime_type": mime_type, "data": image_bytes})

        # --- Generate Content ---
        with st.spinner(f"Analisando {len(page_images_batch)} página(s) com IA Multimodal ({MODEL_NAME})... Esta etapa pode levar alguns minutos."):
            try:
                response = model.generate_content(prompt_parts, stream=False)

                # Process Response (Assume Gemini 1.5 Pro/Latest structure)
                if hasattr(response, 'text'):
                     full_analysis_text = response.text
                elif hasattr(response, 'parts') and response.parts:
                     full_analysis_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    block_message = response.prompt_feedback.block_reason_message or f"Reason code: {block_reason}"
                    full_analysis_text = f"**Análise Bloqueada pela API:** {block_message}"
                    st.error(f"A análise multimodal foi bloqueada pela API: {block_message}")
                else:
                     full_analysis_text = "A API retornou uma resposta vazia ou em formato não esperado."
                     st.warning(f"Resposta inesperada ou vazia da análise: {response}")

            except StopCandidateException as stop_e:
                 full_analysis_text = f"\n\n**Erro de Geração:** A análise foi interrompida prematuramente. Causa provável: {stop_e}. Verifique as políticas de conteúdo ou tente novamente."
                 st.error(f"Erro na Geração Gemini (StopCandidateException): A resposta foi interrompida. Detalhes: {stop_e}")
            except Exception as e:
                 st.error(f"Erro durante a chamada da API Gemini: {str(e)}")
                 full_analysis_text += f"\n\n**Erro Crítico na Análise:** Não foi possível completar a análise devido a um erro inesperado: {str(e)}"

        analysis_output += full_analysis_text

    except Exception as e:
        st.error(f"Erro geral durante a preparação ou análise multimodal: {str(e)}")
        analysis_output += f"\n\n**Erro Crítico:** Falha inesperada: {str(e)}"

    return analysis_output

# --- Callback Function ---
def sync_batch_selection():
    """
    Callback para garantir que a seleção do selectbox seja processada
    antes da próxima renderização completa do script.
    O valor já está em st.session_state.selected_batch devido ao 'key'.
    """
    # print(f"Callback sync_batch_selection: st.session_state.selected_batch is now '{st.session_state.get('selected_batch')}'")
    # Não precisa fazer nada aqui, mas a existência do callback ajuda no fluxo do Streamlit.
    pass


# --- Streamlit Interface ---

st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"""
Envie um arquivo de prova em **PDF**. A ferramenta converterá as páginas em imagens e usará IA multimodal ({MODEL_NAME}) para identificar e analisar as questões **diretamente das imagens**.
Ideal para PDFs escaneados ou onde a extração de texto falha.
**Aviso:** Requer `poppler` instalado. Processamento pode ser mais lento e custoso que análise baseada em texto.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password", help=f"Necessária para usar o {MODEL_NAME}.")

    st.subheader("Opções de Análise")
    # Batch selection populated after upload dynamically below

    st.markdown("---")
    st.markdown(f"""
    ### Como Usar:
    1.  Cole sua chave API do Google Gemini.
    2.  Faça o upload do arquivo PDF.
    3.  Aguarde a conversão (pode levar um tempo).
    4.  Selecione o **batch de páginas** desejado. O botão "Analisar" deve habilitar.
    5.  Clique em "Analisar Batch Selecionado".
    6.  Aguarde a análise multimodal pela IA.
    7.  **Repita os passos 4-6 para outros batches do mesmo PDF.**
    8.  Visualize ou baixe o resultado do batch atual.
    """)
    st.markdown("---")
    st.info("A precisão depende da qualidade da imagem e da capacidade da IA. Verifique os resultados.")
    st.warning("**Dependência Externa:** Requer `poppler` instalado no ambiente de execução.")

# --- Main Area Logic ---

# Initialize session state variables if they don't exist
default_state = {
    'analysis_result': None,
    'error_message': None,
    'pdf_page_images': [], # Stores PIL Images
    'analysis_running': False,
    'uploaded_file_id': None,
    'batch_options': [],
    'selected_batch': None,
    'total_pages': 0,
    'original_filename': None
}
for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.write("## 📄 1. Upload da Prova (PDF)")
uploaded_file = st.file_uploader(
    "Selecione o arquivo PDF",
    type=["pdf"],
    key="file_uploader_pdf_multimodal" # Consistent key
)

# --- Logic after file upload ---
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    if current_file_id != st.session_state.uploaded_file_id:
        st.info(f"Novo arquivo detectado: '{uploaded_file.name}'. Iniciando processamento...")
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.original_filename = uploaded_file.name
        # Resetar tudo relacionado ao arquivo anterior
        for key in default_state:
             if key != 'uploaded_file_id' and key != 'original_filename': # Preservar o novo ID/nome
                  st.session_state[key] = default_state[key]

        # --- Step 1: Convert PDF to Images ---
        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error:
            st.error(f"Falha na Conversão do PDF: {error}")
            st.session_state.error_message = f"Falha na Conversão do PDF: {error}"
            st.session_state.pdf_page_images = []
        elif not images:
            st.warning("Nenhuma imagem foi gerada a partir do PDF.")
            st.session_state.error_message = "Nenhuma imagem foi gerada a partir do PDF."
            st.session_state.pdf_page_images = []
        else:
            st.session_state.pdf_page_images = images
            st.session_state.total_pages = len(images)
            st.success(f"Conversão concluída! {st.session_state.total_pages} páginas prontas para análise.")

            # --- Generate Batch Options ---
            num_batches = math.ceil(st.session_state.total_pages / PAGES_PER_BATCH)
            batch_opts = []
            if st.session_state.total_pages > 1:
                 batch_opts.append("Analisar Todas")
            for i in range(num_batches):
                start_page = i * PAGES_PER_BATCH + 1
                end_page = min((i + 1) * PAGES_PER_BATCH, st.session_state.total_pages)
                if start_page == end_page:
                     batch_opts.append(f"Página {start_page}")
                else:
                     batch_opts.append(f"Páginas {start_page}-{end_page}")

            st.session_state.batch_options = batch_opts
            # Definir seleção inicial (primeiro batch específico, se houver, senão a primeira opção)
            if len(batch_opts) > 1 and "Analisar Todas" in batch_opts:
                 st.session_state.selected_batch = batch_opts[1]
            elif batch_opts:
                 st.session_state.selected_batch = batch_opts[0]
            else:
                 st.session_state.selected_batch = None

            # Rerun para atualizar a UI com as novas opções e seleção padrão
            st.rerun()

# --- Display file details and batch selection UI (if images are ready) ---
if st.session_state.pdf_page_images:
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} processado. {st.session_state.total_pages} páginas prontas.")

    with st.expander("Visualizar Páginas Convertidas (Miniaturas)"):
        max_preview = 10
        cols = st.columns(5)
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
            with cols[i % 5]:
                try:
                    st.image(img, caption=f"Página {i+1}", width=120)
                except Exception as img_disp_err:
                    st.warning(f"Erro exibindo Pág {i+1}: {img_disp_err}")
        if st.session_state.total_pages > max_preview:
            st.markdown(f"*(Pré-visualização limitada às primeiras {max_preview} de {st.session_state.total_pages} páginas)*")

    # --- Batch Selection UI (Sidebar) ---
    with st.sidebar:
         st.subheader("🎯 Selecionar Batch de Páginas")
         if st.session_state.batch_options:
              current_selection = st.session_state.get('selected_batch')
              try:
                   # Apenas encontra o índice da seleção atual para exibição
                   current_index = st.session_state.batch_options.index(current_selection)
              except (ValueError, IndexError):
                   # Se seleção atual inválida ou None, default para índice 0
                   current_index = 0
                   # Não precisa redefinir st.session_state.selected_batch aqui

              # MODIFICAÇÃO: Adicionado on_change=sync_batch_selection
              st.selectbox(
                  "Escolha o intervalo de páginas:",
                  options=st.session_state.batch_options,
                  index=current_index,
                  key='selected_batch', # Vincula ao estado
                  on_change=sync_batch_selection, # Chama o callback na mudança
                  help="Selecione as páginas a serem enviadas para análise pela IA."
              )
         else:
              st.info("Faça upload de um PDF para ver as opções.")

    # --- Analysis Trigger ---
    st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")
    selected_batch_display = st.session_state.get('selected_batch', 'None') # Usa 'None' se vazio

    # Debug (opcional): Verificar o estado antes de renderizar o botão
    # st.write(f"DEBUG (Pre-Button): selected_batch='{st.session_state.get('selected_batch')}', analysis_running={st.session_state.analysis_running}")

    analyze_button = st.button(
         f"Analisar Batch Selecionado ({selected_batch_display})",
         type="primary",
         use_container_width=True,
         disabled=st.session_state.analysis_running or not st.session_state.selected_batch or not st.session_state.pdf_page_images or not api_key
    )

    if analyze_button:
        # Verificações pré-análise
        if not api_key:
            st.error("⚠️ Por favor, insira sua Chave API do Google Gemini na barra lateral.")
            st.stop()
        if not st.session_state.selected_batch:
             st.error("⚠️ Por favor, selecione um batch de páginas válido na barra lateral.")
             st.stop()
        if not st.session_state.pdf_page_images:
             st.error("⚠️ Nenhuma imagem de página encontrada. Faça upload e converta um PDF primeiro.")
             st.stop()

        # --- Iniciar o processo de análise ---
        st.session_state.analysis_running = True
        st.session_state.analysis_result = None # Limpa resultado anterior
        st.session_state.error_message = None   # Limpa erro anterior
        # MODIFICAÇÃO: Removido st.rerun() daqui

# --- Handle Analysis Execution ---
# Este bloco executa se analysis_running for True (definido pelo clique no botão)
if st.session_state.analysis_running:
     # Mostra o spinner durante a preparação e execução
     with st.spinner(f"Preparando e analisando o batch '{st.session_state.selected_batch}'... Isso pode levar um tempo."):
        # --- Determina as páginas ---
        pages_to_process = []
        selected = st.session_state.selected_batch # Usa o valor que DEVE estar correto agora
        all_images = st.session_state.pdf_page_images
        total_pg = st.session_state.total_pages

        if selected == "Analisar Todas":
            pages_to_process = all_images
            st.info(f"Processando todas as {total_pg} páginas...")
        else:
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

                if 0 <= start_index < total_pg and start_index < end_index <= total_pg:
                    pages_to_process = all_images[start_index:end_index]
                    st.info(f"Analisando páginas de {start_page_label} a {end_page_label}...")
                else:
                    st.warning(f"Intervalo inválido ({start_page_label}-{end_page_label}) para {total_pg} páginas.")
                    pages_to_process = []
            except ValueError as parse_e:
                st.error(f"Erro ao interpretar batch '{selected}': {parse_e}.")
                pages_to_process = []

        # --- Executa a análise ---
        analysis_result_text = None
        if pages_to_process:
            analysis_result_text = analyze_pages_with_gemini_multimodal(
                    api_key,
                    pages_to_process,
                )
            st.session_state.analysis_result = analysis_result_text
            if "Erro" in (analysis_result_text or "") or "Bloqueada" in (analysis_result_text or ""):
                 st.session_state.error_message = "Erro durante a análise pela IA. Verifique os detalhes."
        else:
            if not st.session_state.error_message: # Evita sobrescrever erro de parsing
                  st.session_state.error_message = "Nenhuma página selecionada para análise neste batch."

        # --- Análise concluída ---
        st.session_state.analysis_running = False
        # MODIFICAÇÃO: st.rerun() movido para cá
        st.rerun() # Atualiza a UI para mostrar resultado/erro e reabilitar botão

# --- Display Results or Errors ---
if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}")

if st.session_state.analysis_result and not st.session_state.analysis_running:
    st.write(f"## 📊 3. Resultado da Análise Multimodal (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False)

    # --- Download Button ---
    try:
        original_filename_base = "prova"
        if st.session_state.original_filename:
             original_filename_base = os.path.splitext(st.session_state.original_filename)[0]
             original_filename_base = re.sub(r'[^\w\d-]+', '_', original_filename_base)

        batch_suffix = "completo"
        if st.session_state.selected_batch:
             batch_suffix = re.sub(r'[^\w\d-]+', '_', st.session_state.selected_batch).strip('_')

        download_filename = f"analise_multimodal_{original_filename_base}_batch_{batch_suffix}.md"

        st.download_button(
            label=f"📥 Baixar Análise do Batch Atual ({st.session_state.get('selected_batch', 'N/A')}) (Markdown)",
            data=(st.session_state.analysis_result or "").encode('utf-8'), # Garante que seja string antes de encode
            file_name=download_filename,
            mime="text/markdown"
        )
    except Exception as dl_e:
        st.warning(f"Não foi possível gerar o botão de download: {dl_e}")


# --- Footer ---
st.markdown("---")
st.markdown(f"""
**Desenvolvido como ferramenta de auxílio aos estudos.** Utiliza IA Multimodal ({MODEL_NAME}).
*Resultados dependem da qualidade da imagem e da IA. Verifique sempre.*
| Dependências: [Streamlit](https://streamlit.io/), [Google Gemini API](https://ai.google.dev/), [pdf2image](https://github.com/Belval/pdf2image), [Pillow](https://python-pillow.org/) | **Requer Poppler**
""")