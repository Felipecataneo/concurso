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
MODEL_NAME = "gemini-2.5-pro-exp-05-23" # Ou "gemini-pro-vision" se preferir
PAGES_PER_BATCH = 2 # Analisar 2 páginas por vez

# --- Funções Auxiliares ---

@st.cache_data(show_spinner="Convertendo PDF para imagens...")
def convert_pdf_to_images(_pdf_bytes): # Renomeado argumento para evitar shadowing
    """Converts PDF bytes into a list of PIL Image objects."""
    images = []
    error_message = None
    try:
        images = convert_from_bytes(_pdf_bytes, dpi=200, fmt='png', thread_count=os.cpu_count()) # Usar mais threads se disponível
    except PDFInfoNotInstalledError:
        error_message = """
        Erro de Configuração: Poppler não encontrado.
        'pdf2image' requer a instalação do utilitário 'poppler'. Verifique as instruções de instalação para seu sistema.
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
        # Use o modelo mais recente ou o pro-vision se preferir
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
            "*   Indique **CERTO**/**ERRADO** ou a **Alternativa Correta** (ex: **Alternativa C**). Forneça apenas a resposta final aqui.", # Simplificado para clareza do modelo
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
            # Tente salvar como WEBP para eficiência, fallback para PNG se houver erro
            try:
                 img.save(buffer, format="WEBP", lossless=True, quality=90) # Ajuste qualidade se precisar
                 mime_type = "image/webp"
            except Exception as e_webp:
                 st.warning(f"Falha ao salvar como WEBP ({e_webp}), usando PNG.")
                 buffer = io.BytesIO() # Reset buffer
                 img.save(buffer, format="PNG")
                 mime_type = "image/png"

            image_bytes = buffer.getvalue()
            prompt_parts.append({"mime_type": mime_type, "data": image_bytes})

        # --- Generate Content ---
        # Use um spinner específico para a chamada da API
        with st.spinner(f"Analisando {len(page_images_batch)} página(s) com IA Multimodal ({MODEL_NAME})... Esta etapa pode levar alguns minutos."):
            try:
                # Use generate_content para modelos mais recentes
                response = model.generate_content(prompt_parts, stream=False)

                # Process Response (Gemini 1.5 Pro e outros modelos mais recentes)
                if response.parts:
                     full_analysis_text = "".join(part.text for part in response.parts if hasattr(part, "text"))
                # Verificação de bloqueio de segurança
                elif response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason = response.prompt_feedback.block_reason
                    block_message = response.prompt_feedback.block_reason_message or f"Reason code: {block_reason}"
                    full_analysis_text = f"**Análise Bloqueada pela API:** {block_message}"
                    st.error(f"A análise multimodal foi bloqueada pela API: {block_message}")
                # Caso de resposta vazia ou formato inesperado
                else:
                     full_analysis_text = "A API retornou uma resposta vazia ou em formato não esperado."
                     st.warning(f"Resposta inesperada ou vazia da análise: {response}")

            # Captura de exceções específicas da API e genéricas
            except StopCandidateException as stop_e:
                 full_analysis_text = f"\n\n**Erro de Geração:** A análise foi interrompida prematuramente. Causa provável: {stop_e}. Verifique as políticas de conteúdo ou tente novamente."
                 st.error(f"Erro na Geração Gemini (StopCandidateException): A resposta foi interrompida. Detalhes: {stop_e}")
            except Exception as e:
                 st.error(f"Erro durante a chamada da API Gemini: {str(e)}")
                 full_analysis_text += f"\n\n**Erro Crítico na Análise:** Não foi possível completar a análise devido a um erro inesperado: {str(e)}"

        analysis_output += full_analysis_text

    except Exception as e:
        # Captura erros na configuração do genai ou outras exceções gerais
        st.error(f"Erro geral durante a preparação ou análise multimodal: {str(e)}")
        analysis_output += f"\n\n**Erro Crítico:** Falha inesperada: {str(e)}"

    return analysis_output

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
    4.  Selecione o **batch de páginas** desejado.
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
if 'results_by_batch' not in st.session_state:
    st.session_state.results_by_batch = {}  # Armazena resultados de análise por batch

default_state = {
    'analysis_result': None,
    'error_message': None,
    'pdf_page_images': [], # Stores PIL Images
    'analysis_running': False,
    'uploaded_file_id': None,
    'batch_options': [],
    'selected_batch': None,
    'total_pages': 0,
    'original_filename': None,
    'results_by_batch': {} # Inicializado acima
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
    # Use file name and size as a simple ID
    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"

    # Check if it's a NEW file compared to the one stored in session state
    if current_file_id != st.session_state.uploaded_file_id:
        st.info(f"Novo arquivo detectado: '{uploaded_file.name}'. Iniciando processamento...")
        # Reset state for the new file
        st.session_state.uploaded_file_id = current_file_id
        st.session_state.original_filename = uploaded_file.name
        st.session_state.pdf_page_images = []
        st.session_state.analysis_result = None
        st.session_state.error_message = None
        st.session_state.batch_options = []
        st.session_state.selected_batch = None
        st.session_state.analysis_running = False # Ensure analysis is not marked as running
        st.session_state.results_by_batch = {} # Limpa resultados anteriores de outro arquivo

        # --- Step 1: Convert PDF to Images ---
        pdf_bytes = uploaded_file.getvalue()
        # The function call uses the cache if bytes match previous calls
        images, error = convert_pdf_to_images(pdf_bytes)

        if error:
            st.error(f"Falha na Conversão do PDF: {error}")
            st.session_state.error_message = f"Falha na Conversão do PDF: {error}"
            # Stop processing for this file if conversion fails
            st.session_state.pdf_page_images = [] # Ensure image list is empty
        elif not images:
            st.warning("Nenhuma imagem foi gerada a partir do PDF.")
            st.session_state.error_message = "Nenhuma imagem foi gerada a partir do PDF."
            st.session_state.pdf_page_images = []
        else:
            # SUCCESS: Store images and create batch options
            st.session_state.pdf_page_images = images
            st.session_state.total_pages = len(images)
            st.success(f"Conversão concluída! {st.session_state.total_pages} páginas prontas para análise.")

            # --- Generate Batch Options based on Pages ---
            num_batches = math.ceil(st.session_state.total_pages / PAGES_PER_BATCH)
            batch_opts = [] # Start fresh
            # Always add "Analisar Todas" if more than one page
            if st.session_state.total_pages > 1:
                 batch_opts.append("Analisar Todas")

            # Add specific batch options
            for i in range(num_batches):
                start_page = i * PAGES_PER_BATCH + 1
                end_page = min((i + 1) * PAGES_PER_BATCH, st.session_state.total_pages)
                if start_page == end_page:
                     batch_opts.append(f"Página {start_page}")
                else:
                     batch_opts.append(f"Páginas {start_page}-{end_page}")

            st.session_state.batch_options = batch_opts
            # Set default selection (first specific batch, or 'Analisar Todas' if only one page/batch)
            if len(batch_opts) > 1 and "Analisar Todas" in batch_opts:
                 st.session_state.selected_batch = batch_opts[1] # Select first specific batch e.g., "Páginas 1-2"
            elif batch_opts:
                 st.session_state.selected_batch = batch_opts[0] # Select the only option available
            else:
                 st.session_state.selected_batch = None # Should not happen if images were generated

            # Use rerun to update the UI immediately after conversion and batch option generation
            st.rerun()

# --- Display file details and batch selection UI (if images are ready) ---
if st.session_state.pdf_page_images:
    # Display confirmation that file is ready
    file_name_display = f"'{st.session_state.original_filename}'" if st.session_state.original_filename else "Carregado"
    st.success(f"Arquivo {file_name_display} processado. {st.session_state.total_pages} páginas prontas.")

    # --- Expander for Page Thumbnails ---
    with st.expander("Visualizar Páginas Convertidas (Miniaturas)"):
        # Limit preview to avoid excessive rendering time/memory
        max_preview = 10
        cols = st.columns(5) # Display up to 5 thumbnails per row
        for i, img in enumerate(st.session_state.pdf_page_images[:max_preview]):
            with cols[i % 5]:
                try:
                    # Use a smaller width for thumbnails
                    st.image(img, caption=f"Página {i+1}", width=120)
                except Exception as img_disp_err:
                    st.warning(f"Erro exibindo Pág {i+1}: {img_disp_err}")

        if st.session_state.total_pages > max_preview:
            st.markdown(f"*(Pré-visualização limitada às primeiras {max_preview} de {st.session_state.total_pages} páginas)*")

# --- Batch Selection UI (Sidebar) ---
with st.sidebar:
    st.subheader("🎯 Selecionar Batch de Páginas")
    if st.session_state.batch_options:
        # Crie uma chave exclusiva para o selectbox que NÃO seja vinculada ao session_state
        # Isso impede que o Streamlit redefina a seleção automaticamente
        
        # Inicialize selected_batch se ainda não estiver definido
        if st.session_state.selected_batch is None and st.session_state.batch_options:
            if len(st.session_state.batch_options) > 1 and "Analisar Todas" in st.session_state.batch_options:
                st.session_state.selected_batch = st.session_state.batch_options[1]  # Primeiro batch específico
            else:
                st.session_state.selected_batch = st.session_state.batch_options[0]  # Primeira opção
        
        # Encontre o índice atual na lista de opções
        try:
            current_index = st.session_state.batch_options.index(st.session_state.selected_batch)
        except (ValueError, TypeError):
            current_index = 0
            if st.session_state.batch_options:
                st.session_state.selected_batch = st.session_state.batch_options[current_index]
        
        # Criamos uma função callback para atualizar o valor selecionado
        def update_batch_selection():
            # Obtém o valor do widget pelo valor da chave
            selected_value = st.session_state.batch_selector_widget
            # Atualiza session_state apenas se o valor for diferente
            if selected_value != st.session_state.selected_batch:
                st.session_state.selected_batch = selected_value
                
                # Verificar se já existe resultado para esse batch
                if selected_value in st.session_state.results_by_batch:
                    st.session_state.analysis_result = st.session_state.results_by_batch[selected_value]
                    st.success(f"Carregado resultado existente para o batch '{selected_value}'")
        
        # Usa o selectbox com uma chave única e uma função callback
        batch_selection = st.selectbox(
            "Escolha o intervalo de páginas:",
            options=st.session_state.batch_options,
            index=current_index,
            key="batch_selector_widget",
            on_change=update_batch_selection,
            help="Selecione as páginas a serem enviadas para análise pela IA."
        )
        
        # Exibir estado atual para debug (opcional, pode remover em produção)
        st.sidebar.caption(f"Batch selecionado: {st.session_state.selected_batch}")
    else:
        st.info("Aguardando opções de batch...")

    # --- Estado das Análises ---
    if st.session_state.results_by_batch:
        st.sidebar.subheader("📊 Batch(es) Analisado(s)")
        for batch_name in st.session_state.results_by_batch.keys():
            st.sidebar.success(f"✅ {batch_name}")

    # --- Analysis Trigger ---
    st.write("## ⚙️ 2. Iniciar Análise Multimodal do Batch")
    # Ensure selected_batch from state is used for the button label
    selected_batch_display = st.session_state.get('selected_batch', 'Nenhum')
    
    # Verifica se o batch já foi analisado
    batch_already_analyzed = selected_batch_display in st.session_state.results_by_batch
    
    button_text = (
        f"Analisar Batch Selecionado ({selected_batch_display})" 
        if not batch_already_analyzed else 
        f"Atualizar Análise do Batch ({selected_batch_display})"
    )
    
    analyze_button = st.button(
         button_text,
         type="primary",
         use_container_width=True,
         # Disable if analysis running, no batch selected, no images, or no API key
         disabled=st.session_state.analysis_running or not st.session_state.selected_batch or not st.session_state.pdf_page_images or not api_key
    )

    if analyze_button:
        # Double-check preconditions right before starting
        if not api_key:
            st.error("⚠️ Por favor, insira sua Chave API do Google Gemini na barra lateral.")
        elif not st.session_state.selected_batch:
             st.error("⚠️ Por favor, selecione um batch de páginas na barra lateral.")
        elif not st.session_state.pdf_page_images:
             # Should be caught earlier, but good as a safeguard
             st.error("⚠️ Nenhuma imagem de página encontrada. Faça upload e converta um PDF primeiro.")
        else:
            # --- Start Analysis Process ---
            st.session_state.analysis_running = True
            # Limpar resultado atual antes da nova análise
            st.session_state.analysis_result = None
            st.session_state.error_message = None
            st.rerun() # Rerun imediato para mostrar o estado "rodando" e limpar resultados antigos

# --- Handle Analysis Execution (continuação do if analyze_button, mas após o rerun) ---
# This block runs *after* the rerun triggered by setting analysis_running to True
if st.session_state.analysis_running:
     # Show spinner while preparing and running analysis
     with st.spinner(f"Preparando e analisando o batch '{st.session_state.selected_batch}'... Isso pode levar um tempo."):
        # --- Determine Page Images to Analyze Based on Batch ---
        pages_to_process = []
        selected = st.session_state.selected_batch
        all_images = st.session_state.pdf_page_images
        total_pg = st.session_state.total_pages

        if selected == "Analisar Todas":
            pages_to_process = all_images
            st.info(f"Processando todas as {total_pg} páginas...") # Log info is helpful
        else:
            # Use regex to find numbers in the selection string
            nums_str = re.findall(r'\d+', selected)
            try:
                if len(nums_str) == 1: # Single page case like "Página 5"
                    start_page_label = int(nums_str[0])
                    end_page_label = start_page_label
                elif len(nums_str) == 2: # Range case like "Páginas 1-2"
                    start_page_label = int(nums_str[0])
                    end_page_label = int(nums_str[1])
                else:
                    # Should not happen with generated options, but handle defensively
                    raise ValueError(f"Formato de batch inesperado: {selected}")

                # Convert 1-based page labels to 0-based list indices
                start_index = start_page_label - 1
                # Slice goes up to, but not including, end_index, so use end_page_label directly
                end_index = end_page_label

                # Validate indices against available pages
                if 0 <= start_index < total_pg and start_index < end_index <= total_pg:
                    pages_to_process = all_images[start_index:end_index]
                    st.info(f"Analisando páginas de {start_page_label} a {end_page_label}...")
                else:
                    st.warning(f"Intervalo de páginas inválido ({start_page_label}-{end_page_label}) para o total de {total_pg} páginas no batch '{selected}'. Nenhuma página selecionada.")
                    pages_to_process = [] # Ensure it's empty

            except ValueError as parse_e:
                st.error(f"Erro ao interpretar a seleção de batch '{selected}': {parse_e}. Tente selecionar novamente.")
                pages_to_process = [] # Ensure it's empty on parse error

        # --- Proceed with analysis ONLY if pages were successfully selected ---
        if pages_to_process:
            # Call the analysis function (contains its own spinner for the API call)
            analysis_markdown = analyze_pages_with_gemini_multimodal(
                    api_key,
                    pages_to_process, # Pass the list of PIL Image objects for the current batch
                )
            
            # Armazena o resultado no estado da sessão para este batch
            st.session_state.analysis_result = analysis_markdown
            # Salva também no dicionário de resultados por batch
            st.session_state.results_by_batch[selected] = analysis_markdown
            
            # Store potential errors from the analysis function itself if it returns error messages
            if "Erro Crítico" in analysis_markdown or "Análise Bloqueada" in analysis_markdown or "Erro de Geração" in analysis_markdown:
                 st.session_state.error_message = "A análise retornou um erro. Veja detalhes abaixo." # Generic message, details are in analysis_result
                 # Não salva análises com erro no dicionário
                 if selected in st.session_state.results_by_batch:
                     del st.session_state.results_by_batch[selected]

        else:
            # If pages_to_process is empty due to errors above, set an error message
            if not st.session_state.error_message: # Avoid overwriting specific parsing errors
                  st.session_state.error_message = "Nenhuma página foi selecionada para análise neste batch devido a um erro ou intervalo inválido."

        # --- Analysis finished (or failed) ---
        st.session_state.analysis_running = False
        # Rerun again to display results/errors and re-enable the button
        st.rerun()

# --- Display Results or Errors ---
# Display errors prominently if they occurred
if st.session_state.error_message and not st.session_state.analysis_running:
    st.error(f"⚠️ {st.session_state.error_message}")

# Display results if available and analysis is not running
if st.session_state.analysis_result and not st.session_state.analysis_running:
    st.write(f"## 📊 3. Resultado da Análise Multimodal (Batch: {st.session_state.get('selected_batch', 'N/A')})")
    st.markdown(st.session_state.analysis_result, unsafe_allow_html=False) # Prefer False for safety

    # --- Download Button for the current result ---
    try:
        # Sanitize filename parts
        original_filename_base = "prova" # Default
        if st.session_state.original_filename:
             original_filename_base = os.path.splitext(st.session_state.original_filename)[0]
             original_filename_base = re.sub(r'[^\w\d-]+', '_', original_filename_base) # Sanitize

        batch_suffix = "completo"
        if st.session_state.selected_batch:
             batch_suffix = re.sub(r'[^\w\d-]+', '_', st.session_state.selected_batch).strip('_')

        download_filename = f"analise_multimodal_{original_filename_base}_batch_{batch_suffix}.md"

        st.download_button(
            label=f"📥 Baixar Análise do Batch Atual ({st.session_state.get('selected_batch', 'N/A')}) (Markdown)",
            data=st.session_state.analysis_result.encode('utf-8'), # Encode result string to bytes
            file_name=download_filename,
            mime="text/markdown"
        )
        
        # Opção para baixar todas as análises combinadas
        if len(st.session_state.results_by_batch) > 1:
            all_analyses = "\n\n---\n\n".join(st.session_state.results_by_batch.values())
            st.download_button(
                label=f"📥 Baixar TODAS as Análises Combinadas ({len(st.session_state.results_by_batch)} batches) (Markdown)",
                data=all_analyses.encode('utf-8'),
                file_name=f"analise_multimodal_{original_filename_base}_COMPLETA.md",
                mime="text/markdown"
            )
    except Exception as dl_e:
        st.warning(f"Não foi possível gerar o botão de download: {dl_e}")

