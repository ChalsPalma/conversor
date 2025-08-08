import streamlit as st
import os
import tempfile
import shutil
from pdf_extractor import extract_text_from_pdf
from text_chunker import split_text_into_chunks
from embedding_processor import generate_embeddings_and_store
import chromadb
from chromadb.utils import embedding_functions

def main():
    """
    Función principal de la aplicación Streamlit.
    """
    st.set_page_config(
        page_title="PDF a Embeddings Vectoriales",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("📄 Conversor de PDF a Embeddings Vectoriales")
    st.markdown("""
    Esta aplicación convierte documentos PDF en embeddings vectoriales que pueden ser utilizados para:
    - Búsquedas semánticas
    - Sistemas de recomendación
    - Análisis de similitud de documentos
    - Alimentar asistentes de IA
    """)

    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Configuración del modelo de embeddings
    model_options = [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "distilbert-base-nli-stsb-mean-tokens"
    ]
    selected_model = st.sidebar.selectbox(
        "Modelo de Embeddings",
        model_options,
        index=0,
        help="Selecciona el modelo de Sentence-Transformers para generar los embeddings"
    )

    # Configuración de chunking
    chunk_size = st.sidebar.slider(
        "Tamaño del Chunk",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Número máximo de caracteres por chunk"
    )

    chunk_overlap = st.sidebar.slider(
        "Solapamiento entre Chunks",
        min_value=0,
        max_value=min(chunk_size // 2, 500),
        value=100,
        step=25,
        help="Número de caracteres que se solapan entre chunks adyacentes"
    )

    # Nombre de la colección
    collection_name = st.sidebar.text_input(
        "Nombre de la Colección",
        value="pdf_embeddings",
        help="Nombre de la colección en ChromaDB donde se almacenarán los embeddings"
    )

    # Área principal de la aplicación
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📤 Cargar Archivo PDF")
        uploaded_file = st.file_uploader(
            "Selecciona un archivo PDF",
            type=['pdf'],
            help="Sube un archivo PDF para convertir a embeddings vectoriales"
        )

        if uploaded_file is not None:
            # Mostrar información del archivo
            st.success(f"Archivo cargado: {uploaded_file.name}")
            st.info(f"Tamaño: {uploaded_file.size / 1024:.2f} KB")

            # Botón para procesar
            if st.button("🚀 Procesar PDF", type="primary"):
                process_pdf(uploaded_file, selected_model, chunk_size, chunk_overlap, collection_name)

    with col2:
        st.header("📊 Estado del Procesamiento")
        # Esta sección se actualizará durante el procesamiento
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = "Esperando archivo..."
        
        status_placeholder = st.empty()
        status_placeholder.info(st.session_state.processing_status)

    # Sección de búsqueda (solo visible si hay embeddings almacenados)
    st.header("🔍 Búsqueda Semántica")
    search_query = st.text_input(
        "Consulta de búsqueda",
        placeholder="Escribe tu consulta aquí...",
        help="Busca contenido similar en los documentos procesados"
    )

    num_results = st.slider(
        "Número de resultados",
        min_value=1,
        max_value=10,
        value=3,
        help="Cantidad de resultados más similares a mostrar"
    )

    if st.button("🔍 Buscar") and search_query:
        perform_search(search_query, num_results, collection_name)

    # Sección de gestión de colecciones
    st.header("🗂️ Gestión de Colecciones")
    show_collections()

def process_pdf(uploaded_file, model_name, chunk_size, chunk_overlap, collection_name):
    """
    Procesa el archivo PDF cargado y genera embeddings.
    
    Args:
        uploaded_file: Archivo PDF cargado por el usuario
        model_name: Nombre del modelo de embeddings
        chunk_size: Tamaño de los chunks
        chunk_overlap: Solapamiento entre chunks
        collection_name: Nombre de la colección en ChromaDB
    """
    try:
        # Crear un archivo temporal para el PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # Actualizar estado
        st.session_state.processing_status = "Extrayendo texto del PDF..."
        
        # Crear un contenedor para mostrar el progreso
        progress_container = st.container()
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()

        # Paso 1: Extracción de texto
        status_text.text("Extrayendo texto del PDF...")
        progress_bar.progress(25)
        
        text = extract_text_from_pdf(tmp_file_path)
        if not text:
            st.error("No se pudo extraer texto del PDF. Verifica que el archivo no esté dañado.")
            return

        # Paso 2: Chunking
        status_text.text("Dividiendo texto en chunks...")
        progress_bar.progress(50)
        
        chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            st.error("No se pudieron generar chunks del texto extraído.")
            return

        # Paso 3: Generación de embeddings y almacenamiento
        status_text.text(f"Generando embeddings para {len(chunks)} chunks...")
        progress_bar.progress(75)

        # Inicializar ChromaDB
        client = chromadb.PersistentClient(path="./chroma_db")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )

        # Preparar datos
        documents = []
        metadatas = []
        ids = []
        pdf_filename = uploaded_file.name

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "source": pdf_filename,
                "chunk_id": i,
                "page_number": 0,  # TODO: Implementar extracción de número de página
                "chunk_size": len(chunk)
            })
            ids.append(f"{pdf_filename}_{i}")

        # Almacenar en ChromaDB
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        # Completar progreso
        status_text.text("¡Procesamiento completado!")
        progress_bar.progress(100)

        # Mostrar resultados
        st.success(f"✅ PDF procesado exitosamente!")
        st.info(f"📊 Estadísticas del procesamiento:")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Chunks generados", len(chunks))
        with col2:
            st.metric("Caracteres totales", len(text))
        with col3:
            st.metric("Modelo utilizado", model_name)
        with col4:
            st.metric("Colección", collection_name)

        # Mostrar algunos chunks de ejemplo
        st.subheader("📝 Ejemplos de Chunks Generados")
        for i, chunk in enumerate(chunks[:3]):  # Mostrar solo los primeros 3
            with st.expander(f"Chunk {i+1} ({len(chunk)} caracteres)"):
                st.text(chunk)

    except Exception as e:
        st.error(f"Error durante el procesamiento: {str(e)}")
    finally:
        # Limpiar archivo temporal
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)

def perform_search(query, num_results, collection_name):
    """
    Realiza una búsqueda semántica en la colección especificada.
    
    Args:
        query: Consulta de búsqueda
        num_results: Número de resultados a devolver
        collection_name: Nombre de la colección
    """
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        
        # Verificar si la colección existe
        try:
            collection = client.get_collection(name=collection_name)
        except ValueError:
            st.warning(f"La colección '{collection_name}' no existe. Procesa un PDF primero.")
            return

        # Realizar búsqueda
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )

        if results['documents'] and results['documents'][0]:
            st.success(f"Se encontraron {len(results['documents'][0])} resultados:")
            
            for i, (doc, distance, metadata) in enumerate(zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                with st.expander(f"Resultado {i+1} (Similitud: {1-distance:.3f})"):
                    st.text(doc)
                    st.json(metadata)
        else:
            st.info("No se encontraron resultados para la consulta.")

    except Exception as e:
        st.error(f"Error durante la búsqueda: {str(e)}")

def show_collections():
    """
    Muestra información sobre las colecciones existentes en ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(path="./chroma_db")
        collections = client.list_collections()
        
        if collections:
            st.success(f"Colecciones disponibles: {len(collections)}")
            
            for collection in collections:
                with st.expander(f"📁 {collection.name}"):
                    count = collection.count()
                    st.metric("Documentos almacenados", count)
                    
                    if st.button(f"Eliminar {collection.name}", key=f"delete_{collection.name}"):
                        client.delete_collection(collection.name)
                        st.success(f"Colección '{collection.name}' eliminada.")
                        st.experimental_rerun()
        else:
            st.info("No hay colecciones disponibles. Procesa un PDF para crear una.")
            
    except Exception as e:
        st.error(f"Error al acceder a las colecciones: {str(e)}")

if __name__ == "__main__":
    main()

