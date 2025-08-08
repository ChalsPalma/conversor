from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[str]:
    """
    Divide una cadena de texto en chunks utilizando RecursiveCharacterTextSplitter de LangChain.

    Args:
        text (str): La cadena de texto a dividir.
        chunk_size (int): El tamaño máximo de cada chunk.
        chunk_overlap (int): El número de caracteres que se solapan entre chunks adyacentes.

    Returns:
        list[str]: Una lista de cadenas de texto (chunks).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,  # Usa la longitud de caracteres como métrica
        add_start_index=True, # Añade el índice de inicio de cada chunk
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Ejemplo de uso (para pruebas locales)
if __name__ == "__main__":
    sample_text = (
        "El procesamiento del lenguaje natural (PLN) es un campo de la inteligencia artificial que se ocupa de la interacción entre computadoras y el lenguaje humano. "
        "Uno de los desafíos clave en el PLN es la comprensión del contexto y el significado de las palabras. "
        "Las técnicas de chunking son fundamentales para preparar grandes volúmenes de texto para su análisis. "
        "Permiten dividir el texto en unidades más pequeñas que son más manejables para los modelos de aprendizaje automático. "
        "El solapamiento entre chunks ayuda a mantener la continuidad semántica y a evitar la pérdida de información importante en los límites de los chunks. "
        "Esto es especialmente relevante cuando se trabaja con documentos extensos como libros o informes técnicos. "
        "La elección del tamaño del chunk y el solapamiento depende de la aplicación específica y del modelo de embedding utilizado."
    )

    print("\n--- Texto Original ---")
    print(sample_text)

    print("\n--- Chunks Generados (tamaño=100, solapamiento=20) ---")
    chunks = split_text_into_chunks(sample_text, chunk_size=100, chunk_overlap=20)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1} (longitud: {len(chunk)}):\n{chunk}\n")

    print("\n--- Chunks Generados (tamaño=50, solapamiento=10) ---")
    chunks_small = split_text_into_chunks(sample_text, chunk_size=50, chunk_overlap=10)
    for i, chunk in enumerate(chunks_small):
        print(f"Chunk {i+1} (longitud: {len(chunk)}):\n{chunk}\n")


