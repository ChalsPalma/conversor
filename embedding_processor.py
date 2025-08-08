from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os
import fitz

# Asegúrate de que las funciones de extracción y chunking estén disponibles
from pdf_extractor import extract_text_from_pdf
from text_chunker import split_text_into_chunks

def generate_embeddings_and_store(
    pdf_path: str,
    collection_name: str = "pdf_embeddings",
    model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = 500,
    chunk_overlap: int = 100
):
    """
    Extrae texto de un PDF, lo divide en chunks, genera embeddings y los almacena en ChromaDB.

    Args:
        pdf_path (str): La ruta al archivo PDF.
        collection_name (str): El nombre de la colección en ChromaDB donde se almacenarán los embeddings.
        model_name (str): El nombre del modelo de Sentence-Transformers a utilizar.
        chunk_size (int): El tamaño de los chunks de texto.
        chunk_overlap (int): El solapamiento entre chunks de texto.
    """
    print(f"Procesando PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"No se pudo extraer texto de {pdf_path}. Abortando.")
        return

    chunks = split_text_into_chunks(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        print(f"No se generaron chunks para {pdf_path}. Abortando.")
        return

    print(f"Generando embeddings para {len(chunks)} chunks...")
    # Inicializar el cliente de ChromaDB
    # Por defecto, ChromaDB se ejecuta en modo persistente en el directorio actual
    client = chromadb.PersistentClient(path="./chroma_db")

    # Definir la función de embedding para ChromaDB
    # Se usa el modelo de Sentence-Transformers especificado
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)

    # Obtener o crear la colección
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    # Preparar datos para ChromaDB
    documents = []
    metadatas = []
    ids = []
    pdf_filename = os.path.basename(pdf_path)

    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({"source": pdf_filename, "chunk_id": i, "page_number": 0}) # TODO: Mejorar la extracción del número de página
        ids.append(f"{pdf_filename}_chunk_{i}")

    # Añadir los documentos a la colección
    print(f"Añadiendo {len(documents)} documentos a ChromaDB...")
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Embeddings almacenados exitosamente en la colección \'{collection_name}\'.")

    # Ejemplo de búsqueda (para pruebas locales)
    print("\n--- Realizando una búsqueda de ejemplo ---")
    query_text = "inteligencia artificial"
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    print(f"Resultados para la consulta ")
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"  Resultado {i+1} (Distancia: {dist:.4f}):\n    Texto: {doc}\n")
        if results["metadatas"] and results["metadatas"][0] and results["metadatas"][0][i]:
            print(f"    Metadatos: {results['metadatas'][0][i]}\n")

# Ejemplo de uso (para pruebas locales)
if __name__ == "__main__":
    print("Iniciando el bloque de ejemplo...")
    # Asegúrate de tener un archivo PDF de ejemplo llamado 'ejemplo.pdf' en el mismo directorio
    # Puedes usar el generado por pdf_extractor.py o uno propio.
    # Si no existe, crea uno simple para la prueba.
    if not os.path.exists("ejemplo.pdf"):
        print("Creando \'ejemplo.pdf\' para la prueba...")
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "El procesamiento del lenguaje natural (PLN) es un campo de la inteligencia artificial.\n")
            page.insert_text((72, 100), "Se ocupa de la interacción entre computadoras y el lenguaje humano.\n")
            page.insert_text((72, 128), "Los embeddings vectoriales son representaciones numéricas de texto.\n")
            page.insert_text((72, 156), "ChromaDB es una base de datos vectorial para almacenar estos embeddings.")
            doc.save("ejemplo.pdf")
            doc.close()
            print("\'ejemplo.pdf\' creado.")
        except ImportError:
            print("PyMuPDF no está instalado. Por favor, instálalo con: pip install PyMuPDF")
            exit()
        except Exception as e:
            print(f"Error al crear \'ejemplo.pdf\': {e}")
            exit()

    # Importar las funciones necesarias para el ejemplo
    from pdf_extractor import extract_text_from_pdf
    from text_chunker import split_text_into_chunks

    generate_embeddings_and_store("ejemplo.pdf", collection_name="mi_coleccion_prueba")

    # Limpiar el archivo de ejemplo y la base de datos ChromaDB (opcional)
    # import shutil
    # if os.path.exists("ejemplo.pdf"):
    #     os.remove("ejemplo.pdf")
    # if os.path.exists("./chroma_db"):
    #     shutil.rmtree("./chroma_db")


