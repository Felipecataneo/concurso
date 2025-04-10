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
# Use a stable multimodal model
MODEL_NAME = "gemini-2.5-pro-experimental-03-25"
PAGES_PER_BATCH = 2 # Analisar 2 páginas por vez

# --- Funções Auxiliares ---

# Cache image conversion based on file content
@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(pdf_bytes):
    """Converts PDF bytes into a list of PIL Image objects."""
    images = []
    error_message = None
    try:
        # dpi alto melhora a qualidade para OCR, mas aumenta o tamanho/tempo
        images = convert_from_bytes(pdf_bytes, dpi=200, fmt='png', thread_count=2)
    except PDFInfoNotInstalledError:
        error_message = """
        Erro de Configuração: Poppler não encontrado.
        'pdf2image' requer a instalação do utilitário 'poppler'.
        - Linux (apt): sudo apt-get install poppler-utils
        - macOS (brew): brew install poppler
        - Windows: Baixe o Poppler e adicione ao PATH do sistema.
        - Streamlit Cloud: Adicione 'poppler-utils' ao seu packages.txt.
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

def pil_image_to_base64(image, max_width=100):
    """Converts PIL image to base64 string for Streamlit display, resizing."""
    aspect_ratio = image.height / image.width
    new_height = int(max_width * aspect_ratio)
    resized_image = image.resize((max_width, new_height))

    buffered = io.BytesIO()
    resized_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


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
        client = genai.GenerativeModel(
            model_name=MODEL_NAME,
            # Optional: Add safety_settings if needed
            # safety_settings=[...]
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
            "*   Indique **CERTO**/**ERRADO** ou a **Alternativa Correta** (ex: **Alternativa C**).",
            "",
            "### 4. Justificativa Completa",
            "*   Explique detalhadamente o raciocínio. **CRUCIAL:** Se houver contexto, explique COMO ele leva à resposta.",
            "*   Se C/E 'Errado', explique o erro. Se MC, explique por que as outras alternativas estão erradas.",
            "",
            "### 5. Conhecimentos Avaliados",
            "*   Disciplina Principal e Assunto Específico.",
            "",
            "### 6. Dicas e Pegadinhas",
            "*   Tópicos relacionados? Pegadinhas comuns?",
            "```",
            "\n\n**IMPORTANTE:** Analise TODAS as questões visíveis nas imagens a seguir. Se uma questão parecer continuar na próxima página (não incluída neste batch), mencione isso. Apresente as análises das questões na ordem em que aparecem.",
            "\n\n**IMAGENS DAS PÁGINAS PARA ANÁLISE:**\n"
        ]

        # --- FIX: Convert PIL Images to bytes and add as structured parts ---
        for img in page_images_batch:
            # Convert PIL Image to bytes (using WEBP for efficiency)
            buffer = io.BytesIO()
            # Use lossless WEBP to preserve detail, adjust quality if needed
            img.save(buffer, format="WEBP", lossless=True)
            image_bytes = buffer.getvalue()

            # Create the structured part for the image
            image_part = {
                "mime_type": "image/webp",
                "data": image_bytes
            }
            prompt_parts.append(image_part) # Append the dictionary part
            prompt_parts.append("\n---\n") # Separator

        # Remove the trailing separator if it exists
        if prompt_parts[-1] == "\n---\n":
            prompt_parts.pop()
        # --- End Fix ---

        # --- Generate Content ---
        with st.spinner(f"Analisando {len(page_images_batch)} página(s) com IA Multimodal ({MODEL_NAME})..."):
            response = client.generate_content(prompt_parts, stream=False)

        # --- Process Response ---
        if hasattr(response, 'text'):
            full_analysis_text = response.text
        elif hasattr(response, 'parts') and response.parts:
            full_analysis_text = "".join(part.text for part in response.parts if hasattr(part, "text")) # Ensure parts have text
        # Check for blocking feedback *before* assuming empty parts mean no text
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            block_message = response.prompt_feedback.block_reason_message or f"Reason code: {block_reason}"
            full_analysis_text = f"**Análise Bloqueada:** {block_message}"
            st.error(f"A análise multimodal foi bloqueada pela API: {block_message}")
        else:
            # Handle cases where the response might be valid but doesn't contain text
            # (e.g., only function calls, though not expected here)
            full_analysis_text = "Resposta da análise multimodal recebida, mas sem conteúdo de texto extraível."
            st.warning(f"Formato inesperado ou vazio da resposta da análise: {response}")


        analysis_output += full_analysis_text

    # --- FIX: Correct Exception Type Path ---
    except StopCandidateException as stop_e:
    # --- End Fix ---
        st.error(f"Erro na Geração Gemini: A resposta foi interrompida. Detalhes: {stop_e}")
        analysis_output += f"\n\n**Erro de Geração:** A análise foi interrompida ({stop_e}). Pode ser devido a políticas de conteúdo ou outros limites. Tente um batch menor ou verifique o conteúdo."
    except Exception as e:
        # Catch other potential errors during API call or processing
        st.error(f"Erro durante a análise multimodal com a API Gemini: {str(e)}")
        # Include traceback for debugging if running locally/controlled environment
        # import traceback
        # st.error(traceback.format_exc())
        analysis_output += f"\n\n**Erro Crítico na Análise:** Não foi possível completar a análise devido a um erro: {str(e)}"


    return analysis_output

# --- Streamlit Interface (Rest of your code remains the same) ---

st.title("📸 Analisador Multimodal de Provas com IA (Gemini)")
st.markdown(f"""
Envie um arquivo de prova em **PDF**. A ferramenta converterá as páginas em imagens e usará IA multimodal ({MODEL_NAME}) para identificar e analisar as questões **diretamente das imagens**.
Ideal para PDFs escaneados ou onde a extração de texto falha.
**Aviso:** Requer `poppler` instalado. Processamento mais lento e caro que análise de texto.
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configurações")
    api_key = st.text_input("Sua Chave API do Google Gemini", type="password", help=f"Necessária para usar o {MODEL_NAME}.")

    st.subheader("Opções de Análise")
    # Batch selection populated after upload

    st.markdown("---")
    st.markdown(f"""
    ### Como Usar:
    1.  Cole sua chave API do Google Gemini.
    2.  Faça o upload do arquivo PDF.
    3.  **Aguarde a conversão do PDF em imagens.**
    4.  Selecione o **batch de páginas** na barra lateral.
    5.  Clique em "Analisar Batch Selecionado".
    6.  **Aguarde a análise multimodal (pode demorar!).**
    7.  Visualize ou baixe o resultado.
    """)
    st.markdown("---")
    st.info("A precisão depende da qualidade da imagem e da capacidade da IA. Verifique os resultados.")
    st.warning("**Dependência Externa:** Requer `poppler` instalado no ambiente de execução.")

# --- Main Area Logic ---

# Initialize session state variables
if 'analysis_result' not in st.session_state: st.session_state['analysis_result'] = None
if 'error_message' not in st.session_state: st.session_state['error_message'] = None
if 'pdf_page_images' not in st.session_state: st.session_state['pdf_page_images'] = [] # Stores PIL Images
if 'analysis_running' not in st.session_state: st.session_state['analysis_running'] = False
if 'uploaded_file_id' not in st.session_state: st.session_state['uploaded_file_id'] = None
if 'batch_options' not in st.session_state: st.session_state['batch_options'] = []
if 'selected_batch' not in st.session_state: st.session_state['selected_batch'] = None

st.write("## 📄 1. Upload da Prova (PDF)")
uploaded_file = st.file_uploader(
    "Selecione o arquivo PDF",
    type=["pdf"], # Limit to PDF for multimodal processing
    key="file_uploader_pdf_multimodal"
)

# --- Logic after file upload ---
if uploaded_file is not None:
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}-{uploaded_file.type}"
    if current_file_id != st.session_state.get('uploaded_file_id'):
        st.session_state['uploaded_file_id'] = current_file_id
        # Reset states for new file
        st.session_state['pdf_page_images'] = []
        st.session_state['analysis_result'] = None
        st.session_state['error_message'] = None
        st.session_state['batch_options'] = []
        st.session_state['selected_batch'] = None
        st.info("Novo arquivo PDF detectado. Iniciando conversão para imagens...")

        # --- Step 1: Convert PDF to Images ---
        # The @st.cache_data decorator handles the spinner message
        pdf_bytes = uploaded_file.getvalue()
        images, error = convert_pdf_to_images(pdf_bytes)

        if error:
            st.error(f"Falha na Conversão do PDF: {error}")
            st.session_state['error_message'] = f"Falha na Conversão do PDF: {error}"
            # Stop processing if conversion fails
        elif not images:
            st.warning("Nenhuma imagem foi gerada a partir do PDF.")
            st.session_state['error_message'] = "Nenhuma imagem foi gerada a partir do PDF."
            # Stop processing
        else:
            st.session_state['pdf_page_images'] = images
            total_pages = len(images)
            st.success(f"Conversão concluída! {total_pages} páginas convertidas em imagens.")

            # --- Generate Batch Options based on Pages ---
            num_batches = math.ceil(total_pages / PAGES_PER_BATCH)
            batch_opts = ["Analisar Todas"]
            for i in range(num_batches):
                start_page = i * PAGES_PER_BATCH + 1
                end_page = min((i + 1) * PAGES_PER_BATCH, total_pages)
                if start_page == end_page:
                     batch_opts.append(f"Página {start_page}")
                else:
                     batch_opts.append(f"Páginas {start_page}-{end_page}")


            st.session_state['batch_options'] = batch_opts
            # Default selection
            if not st.session_state['selected_batch'] and len(batch_opts) > 1:
                 st.session_state['selected_batch'] = batch_opts[1] # Select first batch
            elif not st.session_state['selected_batch']:
                 st.session_state['selected_batch'] = batch_opts[0] # Select "Analisar Todas"

            # Using st.rerun() here is okay to update the UI state after conversion
            st.rerun()


    # --- Display file details and batch selection UI ---
    if st.session_state.get('pdf_page_images'):
        total_pages = len(st.session_state['pdf_page_images'])
        # Check if file name is available in state before using it
        file_name_display = "Carregado"
        if 'uploaded_file_id' in st.session_state and st.session_state['uploaded_file_id']:
             try:
                 file_name_display = f"'{st.session_state['uploaded_file_id'].split('-')[0]}'" # Extract original name part
             except IndexError:
                 file_name_display = "'Arquivo'" # Fallback

        st.success(f"Arquivo {file_name_display} e convertido. {total_pages} páginas prontas.")

        with st.expander("Visualizar Páginas Convertidas (Miniaturas)"):
             # Limit preview to avoid excessive rendering
             max_preview = 10
             cols = st.columns(5) # Display up to 5 thumbnails per row
             for i, img in enumerate(st.session_state['pdf_page_images'][:max_preview]):
                 with cols[i % 5]:
                     try:
                         st.image(img, caption=f"Página {i+1}", width=100) # Show small thumbnails
                     except Exception as img_disp_err:
                         st.warning(f"Err Pg {i+1}: {img_disp_err}")

             if total_pages > max_preview:
                 st.markdown(f"*(Pré-visualização limitada às primeiras {max_preview} páginas de {total_pages})*")


        # --- Batch Selection UI (Sidebar) ---
        with st.sidebar:
             st.subheader("🎯 Selecionar Batch de Páginas")
             if st.session_state['batch_options']:
                  current_index = 0
                  # Ensure selected_batch exists and is valid before finding index
                  if st.session_state.get('selected_batch') in st.session_state['batch_options']:
                      current_index = st.session_state['batch_options'].index(st.session_state['selected_batch'])
                  else:
                      # Default to first option if current selection is invalid/missing
                      current_index = 0
                      if st.session_state['batch_options']:
                          st.session_state['selected_batch'] = st.session_state['batch_options'][0]


                  # Use a callback to update state immediately on selection change if needed,
                  # otherwise rely on Streamlit's natural flow or the Analyze button press.
                  st.selectbox(
                      "Escolha o intervalo de páginas:",
                      options=st.session_state['batch_options'],
                      index=current_index,
                      key='selected_batch' # Bind directly to session state key
                      # on_change=callback_function # Optional: if immediate action needed on change
                  )
             else:
                  st.info("Faça upload de um PDF para ver as opções.")


        # --- Analysis Trigger ---
        st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")
        analyze_button = st.button(
             f"Analisar Batch Selecionado ({st.session_state.get('selected_batch', 'Nenhum')})",
             type="primary",
             use_container_width=True,
             disabled=st.session_state.get('analysis_running', False) or not st.session_state.get('selected_batch') or not api_key # Also disable if no API key
        )

        if analyze_button:
            # Re-check API key just before analysis
            if not api_key:
                st.error("⚠️ Por favor, insira sua Chave API do Google Gemini na barra lateral.")
            elif not st.session_state.get('selected_batch'):
                 st.error("⚠️ Por favor, selecione um batch de páginas na barra lateral.")
            elif not st.session_state.get('pdf_page_images'):
                 st.error("⚠️ Nenhuma imagem de página encontrada. Faça upload e converta um PDF primeiro.")
            else:
                st.session_state['analysis_running'] = True
                st.session_state['analysis_result'] = None # Clear previous results
                st.session_state['error_message'] = None   # Clear previous errors

                # Use a spinner for the whole analysis preparation and call
                with st.spinner(f"Preparando e analisando o batch '{st.session_state['selected_batch']}'..."):
                    # --- Determine Page Images to Analyze Based on Batch ---
                    pages_to_process = []
                    selected = st.session_state['selected_batch']
                    all_images = st.session_state['pdf_page_images']
                    total_pg = len(all_images)

                    if selected == "Analisar Todas":
                        pages_to_process = all_images
                        st.info(f"Processando todas as {total_pg} páginas...") # Log info
                    else:
                        try:
                            # More robust parsing for "Página X" and "Páginas X-Y"
                            nums_str = re.findall(r'\d+', selected)
                            if len(nums_str) == 1: # Single page case like "Página 5"
                                start_page_label = int(nums_str[0])
                                end_page_label = start_page_label
                            elif len(nums_str) == 2: # Range case like "Páginas 1-2"
                                start_page_label = int(nums_str[0])
                                end_page_label = int(nums_str[1])
                            else:
                                raise ValueError(f"Formato de batch inesperado: {selected}")

                            # Convert to 0-based list indices for slicing
                            start_index = start_page_label - 1
                            end_index = end_page_label # Slice goes up to, but not including, end_index

                            # Ensure indices are within bounds
                            start_index = max(0, start_index)
                            end_index = min(total_pg, end_index) # Cap end index

                            if start_index < end_index and start_index < total_pg:
                                pages_to_process = all_images[start_index:end_index]
                                st.info(f"Processando páginas de {start_page_label} a {end_page_label} (índices {start_index} a {end_index-1})...") # Log info
                            else:
                                st.warning(f"Intervalo de páginas inválido ou fora dos limites para o batch '{selected}'. Nenhuma página selecionada.")

                        except Exception as parse_e:
                            st.error(f"Erro ao processar a seleção de batch '{selected}': {parse_e}. Verifique o formato.")
                            # Don't fallback automatically, let user correct selection or choose 'Analisar Todas'

                    if pages_to_process:
                        # --- Step 2: Analyze with Gemini Multimodal ---
                        # Spinner is now outside, inside function handles API call specifically
                        analysis_markdown = analyze_pages_with_gemini_multimodal(
                                api_key,
                                pages_to_process, # Pass the list of PIL Image objects
                            )
                        st.session_state['analysis_result'] = analysis_markdown
                    else:
                        # Set error message if no pages ended up being processed
                         if not st.session_state.get('error_message'): # Avoid overwriting specific parsing errors
                              st.session_state['error_message'] = "Nenhuma página foi selecionada para análise neste batch."


                st.session_state['analysis_running'] = False
                # Rerun needed to display results/errors and re-enable button
                st.rerun()


# --- Display Results or Errors ---
# Display errors first
if st.session_state.get('error_message'):
    st.error(st.session_state['error_message'])
    # Decide if the error should persist or be cleared after showing once
    # For API/Analysis errors, maybe clear them so user can retry
    # For conversion/config errors, keep them until new file upload
    if not ("Erro de Configuração" in st.session_state['error_message'] or \
            "Falha na Conversão" in st.session_state['error_message'] or \
            "Nenhuma imagem" in st.session_state['error_message']):
         if not st.session_state.get('analysis_running'): # Clear only if not currently running
              # Comment out clearing for now, might be annoying if user wants to see the error
              # st.session_state['error_message'] = None
              pass


# Display results if available
if st.session_state.get('analysis_result'):
    st.write(f"## 📊 3. Resultado da Análise Multimodal (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state['analysis_result'], unsafe_allow_html=False)

    # --- Download Button ---
    try:
        # Attempt to get a base filename, handle potential errors if state is weird
        original_filename_base = "prova-pdf" # Default
        if st.session_state.get('uploaded_file_id'):
             try:
                 original_filename_base = os.path.splitext(st.session_state['uploaded_file_id'].split('-')[0])[0]
             except Exception:
                 pass # Keep default on error

        batch_suffix = re.sub(r'[^\w\d-]+', '_', st.session_state.get('selected_batch', 'completo')).strip('_')
        download_filename = f"analise_multimodal_{original_filename_base}_batch_{batch_suffix}.md"

        st.download_button(
            label=f"📥 Baixar Análise do Batch ({st.session_state.get('selected_batch', 'N/A')}) (Markdown)",
            data=st.session_state['analysis_result'].encode('utf-8'), # Encode to bytes
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
