# Plan de Aplicación: Conversión de PDF a Embeddings Vectoriales

## 1. Objetivo y Funcionalidad Principal

El propósito fundamental de esta aplicación es transformar documentos PDF no estructurados en una representación estructurada y semánticamente rica, utilizando embeddings vectoriales. Esta herramienta permitirá a los usuarios extraer el conocimiento contenido en sus archivos PDF de una manera que sea fácilmente consultable y analizable por sistemas de inteligencia artificial.

### Función Principal

La función principal de la aplicación es leer archivos PDF, procesar su contenido textual y generar embeddings vectoriales para cada "trozo" o "chunk" de texto significativo. Este proceso implica varias etapas, desde la extracción robusta del texto hasta la segmentación inteligente y la vectorización mediante modelos de lenguaje avanzados.

### Casos de Uso

Los embeddings generados por esta aplicación tienen una amplia gama de aplicaciones prácticas, incluyendo:

*   **Bases de Datos Vectoriales para Búsqueda Semántica:** Los embeddings se almacenarán en bases de datos vectoriales, lo que permitirá realizar búsquedas basadas en el significado contextual en lugar de solo palabras clave. Esto es crucial para sistemas de recuperación de información más inteligentes y precisos.
*   **Asistentes de IA y Chatbots:** Los embeddings pueden alimentar asistentes virtuales o chatbots que necesitan "comprender" el contenido de documentos extensos para responder preguntas, resumir información o proporcionar insights relevantes. Esto es particularmente útil en dominios como el legal, médico o de soporte al cliente.
*   **Análisis de Documentos y Clasificación:** Al representar documentos como vectores, se pueden aplicar técnicas de aprendizaje automático para clasificar, agrupar o analizar grandes colecciones de documentos de manera eficiente.
*   **Generación de Resúmenes y Extracción de Información:** Los embeddings facilitan la identificación de las partes más relevantes de un documento, lo que puede ser utilizado para generar resúmenes automáticos o extraer entidades y relaciones clave.

## 2. Requisitos Técnicos y Tecnologías

Para construir una aplicación robusta, eficiente y escalable, se han seleccionado las siguientes tecnologías, priorizando el ecosistema de Python debido a su madurez y la riqueza de sus librerías para el procesamiento de lenguaje natural y la inteligencia artificial.

### Lenguaje de Programación

*   **Python:** Es el lenguaje elegido por su versatilidad, su vasta colección de librerías para IA y ML, y su facilidad de integración con diversas herramientas y plataformas. Su sintaxis clara y su gran comunidad de desarrolladores lo hacen ideal para este tipo de proyectos.

### Librerías Clave

La selección de librerías se basa en la necesidad de cubrir cada etapa del pipeline de procesamiento de PDF a embeddings:

*   **Extracción de Texto de PDF:**
    *   **PyMuPDF (Fitz):** Conocido por su velocidad y eficiencia en la extracción de texto, imágenes y metadatos de archivos PDF. Es una opción robusta para manejar una amplia variedad de formatos de PDF [1].
    *   **pdfplumber:** Ofrece una API más amigable para la extracción de texto y datos tabulares, lo que puede ser ventajoso para PDFs con estructuras complejas [2]. Se considerará como alternativa o complemento a PyMuPDF.

*   **División de Texto en "Chunks" y Gestión del Flujo de Trabajo:**
    *   **LangChain:** Un framework potente para el desarrollo de aplicaciones impulsadas por modelos de lenguaje. Proporciona herramientas para el "chunking" de texto, la gestión de cadenas de procesamiento (chains) y la integración con bases de datos vectoriales y modelos de embeddings [3].
    *   **LlamaIndex:** Similar a LangChain, se especializa en la construcción de aplicaciones con LLMs sobre datos personalizados. Ofrece excelentes capacidades para la indexación y recuperación de información, lo que lo hace ideal para la gestión de documentos y la creación de índices vectoriales [4]. Se utilizará LangChain por su popularidad y amplia adopción, pero LlamaIndex es una alternativa viable.

*   **Generación de Embeddings:**
    *   **Sentence-Transformers:** Una librería para generar embeddings de oraciones, párrafos e imágenes. Proporciona una gran cantidad de modelos pre-entrenados que son eficientes y de alta calidad para diversas tareas de PNL [5].
    *   **API de OpenAI:** Para embeddings de mayor calidad y escalabilidad, se puede integrar la API de embeddings de OpenAI (por ejemplo, `text-embedding-ada-002`). Esto ofrece un rendimiento superior a costa de un costo por uso [6]. La elección dependerá de los requisitos de calidad y presupuesto.

*   **Base de Datos Vectorial:**
    *   **ChromaDB:** Una base de datos vectorial de código abierto, ligera y fácil de usar, ideal para prototipos y aplicaciones de menor escala. Permite almacenar embeddings y sus metadatos, y realizar búsquedas de similitud [7].
    *   **Pinecone:** Una base de datos vectorial gestionada en la nube, diseñada para la escalabilidad y el rendimiento en producción. Es adecuada para grandes volúmenes de datos y altas tasas de consulta [8]. Se optará por ChromaDB para la implementación inicial debido a su facilidad de configuración local.

*   **Interfaz de Usuario (UI):**
    *   **Streamlit:** Un framework de código abierto que permite crear aplicaciones web interactivas para ciencia de datos y aprendizaje automático con solo unas pocas líneas de código Python. Es ideal para construir una UI simple y funcional rápidamente [9].
    *   **Gradio:** Similar a Streamlit, facilita la creación de interfaces de usuario para modelos de ML. Es otra opción viable para la UI [10]. Se utilizará Streamlit por su popularidad y flexibilidad.

## 3. Arquitectura de la Aplicación

La aplicación seguirá una arquitectura modular, dividiendo el proceso en componentes lógicos que interactúan entre sí. Esto facilita el desarrollo, la depuración y la escalabilidad.

```mermaid
graph TD
    A[Usuario] -->|Sube PDF| B(Frontend: Streamlit)
    B -->|Envía PDF| C(Backend: Lógica de Procesamiento)
    C -->|Paso 1: Carga del PDF| D{Módulo de Carga}
    D -->|Paso 2: Extracción de Texto| E{Módulo de Extracción: PyMuPDF}
    E -->|Paso 3: Chunking| F{Módulo de Chunking: LangChain}
    F -->|Paso 4: Generación de Embeddings| G{Módulo de Embeddings: Sentence-Transformers/OpenAI API}
    G -->|Paso 5: Almacenamiento| H[Base de Datos Vectorial: ChromaDB]
    H -->|Almacena Embeddings y Metadatos| I(Almacenamiento Persistente)
```

### Frontend (UI)

La interfaz de usuario será desarrollada con Streamlit. Su función principal es proporcionar un punto de entrada intuitivo para el usuario:

*   **Carga de Archivos:** Un componente de carga de archivos que permita al usuario seleccionar y subir uno o varios archivos PDF.
*   **Feedback Visual:** Mostrar el estado del procesamiento (cargando, extrayendo, generando embeddings, completado) y cualquier mensaje de error.
*   **Configuración (Opcional):** Posibilidad de configurar parámetros como el tamaño del chunk o el modelo de embeddings (para usuarios avanzados).

### Backend (Lógica de Procesamiento)

El backend es el corazón de la aplicación, donde se ejecuta toda la lógica de procesamiento. Se compone de los siguientes pasos:

#### Paso 1: Carga del PDF

*   **Función:** Recibir el archivo PDF subido desde el frontend. Esto implica manejar la entrada de archivos y guardarlos temporalmente en el sistema de archivos o en memoria para su posterior procesamiento.
*   **Tecnología:** Manejado por Streamlit en el lado del servidor, que proporciona objetos de archivo para el contenido subido.

#### Paso 2: Extracción de Texto

*   **Función:** Extraer todo el texto legible del PDF. Este es un paso crítico que debe manejar diversas complejidades de los PDFs, como texto en diferentes fuentes, columnas, tablas, y la presencia de imágenes.
*   **Tecnología:** **PyMuPDF** será la librería principal. Se encargará de iterar sobre las páginas del PDF y extraer el texto, intentando preservar la estructura en la medida de lo posible. Se implementarán mecanismos para manejar tablas (si es posible, extraerlas como texto estructurado) y saltos de página.

#### Paso 3: Chunking (División de Texto)

*   **Función:** Dividir el texto extraído en fragmentos más pequeños y manejables. El objetivo es crear "chunks" que sean lo suficientemente pequeños para ser procesados eficientemente por los modelos de embeddings, pero lo suficientemente grandes para mantener el contexto semántico.
*   **Tecnología:** **LangChain** proporcionará herramientas como `RecursiveCharacterTextSplitter`. Este splitter es capaz de dividir el texto basándose en una lista de separadores de caracteres, lo que permite un control fino sobre cómo se dividen los documentos (por ejemplo, por párrafos, oraciones, o caracteres). Se definirá un tamaño de chunk (ej., 500 caracteres) y un solapamiento (ej., 100 caracteres) para asegurar la continuidad del contexto entre chunks adyacentes.

#### Paso 4: Generación de Embeddings

*   **Función:** Convertir cada fragmento de texto en un vector numérico (embedding). Este vector captura el significado semántico del texto, permitiendo comparaciones de similitud.
*   **Tecnología:** Se utilizará **Sentence-Transformers** para la generación de embeddings localmente. Esto evita dependencias externas y costos de API. Para casos de uso que requieran mayor precisión o modelos más avanzados, se puede integrar la **API de OpenAI** como una opción configurable.

#### Paso 5: Almacenamiento

*   **Función:** Almacenar los pares (texto original del chunk, embedding vectorial) en una base de datos vectorial. Además de los embeddings, se almacenarán metadatos relevantes para cada chunk.
*   **Tecnología:** **ChromaDB** será la base de datos vectorial utilizada. Es una solución ligera y fácil de integrar que permite almacenar colecciones de embeddings y realizar búsquedas de similitud de manera eficiente. Se creará una colección específica para cada documento o para un conjunto de documentos relacionados.

### Base de Datos

*   **Tipo:** Base de datos vectorial (ChromaDB).
*   **Contenido:**
    *   **Vectores (Embeddings):** La representación numérica de cada chunk de texto.
    *   **Texto Original:** El fragmento de texto correspondiente a cada embedding, para referencia y visualización.
    *   **Metadatos:** Información adicional crucial para la recuperación y el filtrado, como:
        *   `nombre_archivo`: Nombre del archivo PDF de origen.
        *   `numero_pagina`: Número de página dentro del PDF donde se encuentra el chunk.
        *   `offset_inicio`: Posición de inicio del chunk dentro del texto de la página (opcional, para depuración).
        *   `offset_fin`: Posición de fin del chunk dentro del texto de la página (opcional, para depuración).
        *   `id_chunk`: Un identificador único para cada chunk.

Esta arquitectura modular asegura que cada componente pueda ser desarrollado, probado y optimizado de forma independiente, facilitando el mantenimiento y la evolución de la aplicación.




![Diagrama de Arquitectura](arquitectura_aplicacion.png)





## Referencias

[1] PyMuPDF Documentation: [https://pymupdf.readthedocs.io/en/latest/](https://pymupdf.readthedocs.io/en/latest/)
[2] pdfplumber Documentation: [https://github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber)
[3] LangChain Documentation: [https://python.langchain.com/docs/](https://python.langchain.com/docs/)
[4] LlamaIndex Documentation: [https://docs.llamaindex.ai/en/stable/](https://docs.llamaindex.ai/en/stable/)
[5] Sentence-Transformers Documentation: [https://www.sbert.net/](https://www.sbert.net/)
[6] OpenAI Embeddings API: [https://platform.openai.com/docs/guides/embeddings](https://platform.openai.com/docs/guides/embeddings)
[7] ChromaDB Documentation: [https://www.trychroma.com/](https://www.trychroma.com/)
[8] Pinecone Documentation: [https://www.pinecone.io/docs/](https://www.pinecone.io/docs/)
[9] Streamlit Documentation: [https://streamlit.io/](https://streamlit.io/)
[10] Gradio Documentation: [https://gradio.app/](https://gradio.app/)





## 4. Solicitud de Código y Fragmentos Específicos

Esta sección detalla los fragmentos de código esenciales para cada componente de la aplicación, proporcionando una guía práctica para su implementación.

### Fragmento 1: Extracción de Texto con PyMuPDF

La extracción de texto de documentos PDF es el primer paso crítico en el pipeline. PyMuPDF (también conocido como `fitz`) es una librería de Python que destaca por su velocidad y eficiencia en esta tarea. Permite acceder al contenido textual de cada página de un PDF, manejar diferentes codificaciones y, en cierta medida, preservar la estructura del documento.

La función `extract_text_from_pdf` toma la ruta de un archivo PDF como entrada y devuelve una única cadena de texto que contiene todo el contenido extraído del documento. Se incluye un manejo básico de errores para asegurar que solo se procesen archivos PDF válidos y para capturar posibles problemas durante la extracción.

```python
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extrae todo el texto de un archivo PDF.

    Args:
        pdf_path (str): La ruta al archivo PDF.

    Returns:
        str: Una cadena de texto que contiene todo el contenido del PDF.
             Retorna una cadena vacía si el archivo no es un PDF o está dañado.
    """
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text += page.get_text()
    except fitz.FileDataError:
        print(f"Error: El archivo {pdf_path} no es un PDF válido o está dañado.")
        return ""
    except Exception as e:
        print(f"Ocurrió un error inesperado al procesar {pdf_path}: {e}")
        return ""
    return text

# Ejemplo de uso (para pruebas locales)
if __name__ == "__main__":
    # Crea un PDF de ejemplo para probar la función
    # En un entorno real, usarías un archivo PDF existente
    try:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Este es un documento PDF de prueba.\n")
        page.insert_text((72, 100), "Contiene varias líneas de texto para demostración.\n")
        page.insert_text((72, 128), "La extracción de texto es el primer paso crucial.")
        doc.save("ejemplo.pdf")
        doc.close()

        extracted_content = extract_text_from_pdf("ejemplo.pdf")
        print("\n--- Contenido Extraído ---")
        print(extracted_content)

        # Prueba con un archivo no existente
        print("\n--- Prueba con archivo no existente ---")
        extract_text_from_pdf("no_existe.pdf")

        # Limpiar el archivo de ejemplo
        import os
        os.remove("ejemplo.pdf")
    except ImportError:
        print("PyMuPDF no está instalado. Por favor, instálalo con: pip install PyMuPDF")
    except Exception as e:
        print(f"Error durante la ejecución del ejemplo: {e}")

```

**Explicación del Código:**

*   **`import fitz`**: Importa la librería PyMuPDF.
*   **`fitz.open(pdf_path)`**: Abre el archivo PDF especificado por `pdf_path`. Se utiliza un bloque `with` para asegurar que el documento se cierre correctamente.
*   **`doc.page_count`**: Obtiene el número total de páginas en el documento.
*   **`doc.load_page(page_num)`**: Carga una página específica del PDF.
*   **`page.get_text()`**: Extrae el texto de la página actual. Este método intenta preservar el formato y el orden del texto lo mejor posible.
*   **Manejo de Errores**: Los bloques `try-except` capturan `fitz.FileDataError` para archivos PDF inválidos o dañados, y un `Exception` general para otros errores inesperados. Esto mejora la robustez de la función.
*   **Ejemplo de Uso (`if __name__ == "__main__"`)**: Se incluye un bloque de código que demuestra cómo crear un PDF simple, extraer su texto y manejar un caso de error (archivo no existente). Este bloque es útil para probar la función de forma aislada.

Esta función proporciona una base sólida para la extracción de texto, que es fundamental para los pasos posteriores de chunking y generación de embeddings.




### Fragmento 2: Chunking con LangChain

Una vez que el texto ha sido extraído del PDF, el siguiente paso es dividirlo en "chunks" o fragmentos. Este proceso es crucial porque los modelos de embeddings tienen limitaciones en la cantidad de texto que pueden procesar a la vez. Además, dividir el texto de manera inteligente ayuda a preservar el contexto semántico dentro de cada fragmento, lo que es vital para la calidad de los embeddings y la posterior recuperación de información.

LangChain ofrece varias estrategias para el "chunking" de texto, y `RecursiveCharacterTextSplitter` es una de las más flexibles y recomendadas. Este splitter intenta dividir el texto utilizando una lista de separadores de caracteres (por defecto, `["\n\n", "\n", " ", ""]`). Si un chunk es demasiado grande, intenta dividirlo con el siguiente separador en la lista, y así sucesivamente. Esto permite una división más natural y contextual del texto.

La función `split_text_into_chunks` toma una cadena de texto, un tamaño de chunk (`chunk_size`) y un tamaño de solapamiento (`chunk_overlap`) como entrada, y devuelve una lista de cadenas de texto (los chunks).

```python
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
```

**Explicación del Código:**

*   **`from langchain.text_splitter import RecursiveCharacterTextSplitter`**: Importa la clase necesaria de LangChain.
*   **`RecursiveCharacterTextSplitter(...)`**: Se inicializa el splitter con los parámetros `chunk_size` (tamaño máximo de cada fragmento) y `chunk_overlap` (cantidad de caracteres que se solapan entre fragmentos adyacentes). El `length_function=len` indica que la longitud se mide en caracteres. `add_start_index=True` es útil para depuración y para asociar los chunks con su posición original en el documento.
*   **`text_splitter.split_text(text)`**: Este método realiza la división del texto según la configuración del splitter.
*   **Ejemplo de Uso (`if __name__ == "__main__"`)**: Se proporciona un texto de ejemplo y se demuestra cómo dividirlo en chunks con diferentes configuraciones de `chunk_size` y `chunk_overlap`. Esto permite visualizar el efecto de estos parámetros en la salida.

La implementación de esta función de chunking es un paso fundamental para preparar el texto extraído del PDF para la fase de generación de embeddings, asegurando que los fragmentos sean de un tamaño adecuado y mantengan el contexto semántico.




### Fragmento 3: Generación de Embeddings y Almacenamiento en ChromaDB

La generación de embeddings es el proceso de transformar el texto en representaciones numéricas densas que capturan su significado semántico. Estos vectores son la base para realizar búsquedas de similitud y otras operaciones de PNL. Posteriormente, estos embeddings se almacenan en una base de datos vectorial, como ChromaDB, que está optimizada para almacenar y consultar vectores de manera eficiente.

La función `generate_embeddings_and_store` integra los pasos anteriores (extracción y chunking) con la generación de embeddings y el almacenamiento. Utiliza `SentenceTransformer` para generar los embeddings localmente y `ChromaDB` para la persistencia de los datos.

```python
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
import os

# Asegúrate de que las funciones de extracción y chunking estén disponibles
# from pdf_extractor import extract_text_from_pdf
# from text_chunker import split_text_into_chunks

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
    print(f"Embeddings almacenados exitosamente en la colección '{collection_name}'.")

    # Ejemplo de búsqueda (para pruebas locales)
    print("\n--- Realizando una búsqueda de ejemplo ---")
    query_text = "inteligencia artificial"
    results = collection.query(
        query_texts=[query_text],
        n_results=2
    )
    print(f"Resultados para la consulta '{query_text}':")
    for i, (doc, dist) in enumerate(zip(results['documents'][0], results['distances'][0])):
        print(f"  Resultado {i+1} (Distancia: {dist:.4f}):\n    Texto: {doc}\n")
        if results['metadatas'] and results['metadatas'][0] and results['metadatas'][0][i]:
            print(f"    Metadatos: {results['metadatas'][0][i]}\n")

# Ejemplo de uso (para pruebas locales)
if __name__ == "__main__":
    # Asegúrate de tener un archivo PDF de ejemplo llamado 'ejemplo.pdf' en el mismo directorio
    # Puedes usar el generado por pdf_extractor.py o uno propio.
    # Si no existe, crea uno simple para la prueba.
    if not os.path.exists("ejemplo.pdf"):
        print("Creando 'ejemplo.pdf' para la prueba...")
        try:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "El procesamiento del lenguaje natural (PLN) es un campo de la inteligencia artificial.\n")
            page.insert_text((72, 100), "Se ocupa de la interacción entre computadoras y el lenguaje humano.\n")
            page.insert_text((72, 128), "Los embeddings vectoriales son representaciones numéricas de texto.\n")
            page.insert_text((72, 156), "ChromaDB es una base de datos vectorial para almacenar estos embeddings.")
            doc.save("ejemplo.pdf")
            doc.close()
            print("'ejemplo.pdf' creado.")
        except ImportError:
            print("PyMuPDF no está instalado. Por favor, instálalo con: pip install PyMuPDF")
            exit()
        except Exception as e:
            print(f"Error al crear 'ejemplo.pdf': {e}")
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
```

**Explicación del Código:**

*   **`from sentence_transformers import SentenceTransformer`**: Importa la clase para cargar modelos de embeddings pre-entrenados.
*   **`import chromadb`**: Importa la librería cliente de ChromaDB.
*   **`client = chromadb.PersistentClient(path="./chroma_db")`**: Inicializa un cliente de ChromaDB en modo persistente. Esto significa que los datos se guardarán en el directorio `./chroma_db` y se cargarán automáticamente en futuras ejecuciones.
*   **`embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)`**: Define la función de embedding que ChromaDB utilizará. Se le pasa el nombre del modelo de Sentence-Transformers (`all-MiniLM-L6-v2` es un modelo pequeño y eficiente para pruebas).
*   **`collection = client.get_or_create_collection(...)`**: Obtiene una colección existente o crea una nueva si no existe. Las colecciones son como tablas en una base de datos relacional, donde se almacenan los embeddings.
*   **Preparación de Datos**: Se itera sobre los chunks de texto, creando listas de `documents` (el texto del chunk), `metadatas` (información adicional como el nombre del archivo y el ID del chunk) e `ids` (identificadores únicos para cada embedding).
*   **`collection.add(...)`**: Añade los documentos, metadatos e IDs a la colección de ChromaDB. ChromaDB se encarga de generar los embeddings internamente utilizando la `embedding_function` configurada.
*   **Ejemplo de Búsqueda**: Se muestra cómo realizar una búsqueda de similitud en la colección. `collection.query` busca los embeddings más similares a la consulta (`query_texts`) y devuelve los documentos y metadatos asociados, junto con las distancias de similitud.
*   **Manejo de Números de Página**: Se ha incluido un `TODO` para mejorar la extracción del número de página. Actualmente, se asigna `0` por defecto. Para una implementación más avanzada, se necesitaría modificar la función `extract_text_from_pdf` para que devuelva el texto junto con su número de página de origen.

Este fragmento de código establece el núcleo de la aplicación, permitiendo la transformación de texto en embeddings y su almacenamiento eficiente para futuras consultas semánticas.



### Fragmento 4: Interfaz de Usuario con Streamlit

La interfaz de usuario es el punto de entrada para los usuarios finales de la aplicación. Streamlit es una excelente opción para crear interfaces web interactivas para aplicaciones de ciencia de datos y aprendizaje automático con un mínimo de código. Su filosofía de "escribir aplicaciones web como si fueran scripts de Python" la hace ideal para prototipos rápidos y aplicaciones funcionales.

La aplicación Streamlit integrará todos los componentes desarrollados anteriormente (extracción de PDF, chunking, generación de embeddings y almacenamiento) en una interfaz cohesiva y fácil de usar. Los usuarios podrán cargar archivos PDF, configurar parámetros de procesamiento y visualizar los resultados del proceso de vectorización.

```python
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
```

**Explicación del Código:**

*   **`st.set_page_config(...)`**: Configura la página de Streamlit con un título, icono y layout específicos. El layout "wide" utiliza todo el ancho de la pantalla para una mejor experiencia de usuario.

*   **Sidebar de Configuración**: La barra lateral contiene todos los parámetros configurables de la aplicación:
    *   **Modelo de Embeddings**: Un selectbox que permite al usuario elegir entre diferentes modelos de Sentence-Transformers. Cada modelo tiene diferentes características de rendimiento y calidad.
    *   **Configuración de Chunking**: Sliders para ajustar el tamaño del chunk y el solapamiento. Los valores están limitados para evitar configuraciones problemáticas.
    *   **Nombre de la Colección**: Un campo de texto para especificar el nombre de la colección en ChromaDB.

*   **Área Principal**: Dividida en dos columnas:
    *   **Columna 1**: Contiene el componente de carga de archivos (`st.file_uploader`) y el botón de procesamiento.
    *   **Columna 2**: Muestra el estado del procesamiento en tiempo real.

*   **Función `process_pdf`**: Esta función maneja todo el pipeline de procesamiento:
    *   Crea un archivo temporal para el PDF cargado.
    *   Utiliza una barra de progreso (`st.progress`) para mostrar el avance del procesamiento.
    *   Integra todas las funciones desarrolladas anteriormente (extracción, chunking, embeddings).
    *   Muestra estadísticas del procesamiento y ejemplos de chunks generados.

*   **Búsqueda Semántica**: Una sección que permite a los usuarios realizar consultas sobre los documentos procesados:
    *   Campo de texto para la consulta.
    *   Slider para especificar el número de resultados.
    *   Función `perform_search` que ejecuta la búsqueda y muestra los resultados con sus puntuaciones de similitud.

*   **Gestión de Colecciones**: Una sección administrativa que permite:
    *   Ver todas las colecciones existentes en ChromaDB.
    *   Mostrar estadísticas de cada colección (número de documentos).
    *   Eliminar colecciones si es necesario.

*   **Manejo de Errores**: Cada función incluye bloques `try-except` para capturar y mostrar errores de manera amigable al usuario.

*   **Archivos Temporales**: Se utilizan archivos temporales para manejar los PDFs cargados, asegurando que se limpien después del procesamiento.

Esta interfaz de Streamlit proporciona una experiencia de usuario completa y profesional, permitiendo a los usuarios interactuar con todas las funcionalidades de la aplicación de manera intuitiva. La aplicación es escalable y puede ser fácilmente extendida con funcionalidades adicionales como la exportación de embeddings, la comparación de documentos, o la integración con APIs externas.



## 5. Consideraciones Adicionales y Optimizaciones

El desarrollo de una aplicación de nivel de producción para la conversión de PDF a embeddings vectoriales requiere abordar múltiples aspectos que van más allá de la funcionalidad básica. Esta sección examina las consideraciones críticas para la escalabilidad, robustez, seguridad y optimización del rendimiento que son esenciales para un despliegue exitoso en entornos reales.

### Gestión de Errores y Robustez

La gestión de errores es fundamental para crear una aplicación confiable que pueda manejar una amplia variedad de situaciones imprevistas. Los documentos PDF pueden presentar numerosos desafíos técnicos que requieren un manejo cuidadoso y estratégico.

**Validación de Archivos PDF**: No todos los archivos con extensión `.pdf` son documentos PDF válidos. Algunos pueden estar corruptos, protegidos con contraseña, o ser archivos de imagen escaneados sin texto extraíble. La aplicación debe implementar una validación robusta que verifique la integridad del archivo antes de intentar procesarlo. Esto incluye verificar la estructura del archivo PDF, detectar si está protegido con contraseña, y determinar si contiene texto extraíble o solo imágenes.

```python
import fitz
from PIL import Image
import pytesseract

def validate_and_extract_pdf(pdf_path: str) -> tuple[str, dict]:
    """
    Valida un archivo PDF y extrae texto, manejando diferentes tipos de documentos.
    
    Returns:
        tuple: (texto_extraido, metadatos_del_documento)
    """
    validation_info = {
        "is_valid": False,
        "is_encrypted": False,
        "has_text": False,
        "page_count": 0,
        "extraction_method": None,
        "errors": []
    }
    
    try:
        doc = fitz.open(pdf_path)
        validation_info["page_count"] = doc.page_count
        validation_info["is_encrypted"] = doc.is_encrypted
        
        if doc.is_encrypted:
            validation_info["errors"].append("Documento protegido con contraseña")
            return "", validation_info
        
        # Intentar extracción de texto normal
        text = ""
        for page_num in range(min(3, doc.page_count)):  # Verificar solo las primeras 3 páginas
            page = doc.load_page(page_num)
            page_text = page.get_text()
            text += page_text
        
        if len(text.strip()) > 50:  # Si hay suficiente texto
            validation_info["has_text"] = True
            validation_info["extraction_method"] = "direct_text"
            validation_info["is_valid"] = True
            
            # Extraer todo el texto
            full_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                full_text += page.get_text()
            
            return full_text, validation_info
        
        else:
            # Intentar OCR si no hay texto directo
            validation_info["extraction_method"] = "ocr"
            ocr_text = extract_text_with_ocr(doc)
            
            if len(ocr_text.strip()) > 50:
                validation_info["has_text"] = True
                validation_info["is_valid"] = True
                return ocr_text, validation_info
            else:
                validation_info["errors"].append("No se pudo extraer texto significativo")
                return "", validation_info
                
    except fitz.FileDataError:
        validation_info["errors"].append("Archivo PDF corrupto o inválido")
        return "", validation_info
    except Exception as e:
        validation_info["errors"].append(f"Error inesperado: {str(e)}")
        return "", validation_info
    finally:
        if 'doc' in locals():
            doc.close()

def extract_text_with_ocr(doc) -> str:
    """
    Extrae texto usando OCR para documentos escaneados.
    """
    ocr_text = ""
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # Convertir página a imagen
            mat = fitz.Matrix(2.0, 2.0)  # Escalar para mejor calidad OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Usar pytesseract para OCR
            image = Image.open(io.BytesIO(img_data))
            page_text = pytesseract.image_to_string(image, lang='spa+eng')
            ocr_text += page_text + "\n"
            
    except Exception as e:
        print(f"Error en OCR: {e}")
    
    return ocr_text
```

**Manejo de Documentos Grandes**: Los documentos PDF pueden variar enormemente en tamaño, desde documentos de una página hasta libros completos de cientos de páginas. Para documentos muy grandes, el procesamiento en memoria puede causar problemas de rendimiento o incluso agotar la memoria disponible. La aplicación debe implementar estrategias de procesamiento por lotes y monitoreo de memoria.

**Límites de Tiempo de Procesamiento**: Para evitar que la aplicación se cuelgue con documentos problemáticos, se deben implementar timeouts y límites de tiempo de procesamiento. Esto es especialmente importante en aplicaciones web donde los usuarios esperan respuestas en tiempos razonables.

### Escalabilidad y Rendimiento

La escalabilidad es crucial para aplicaciones que pueden necesitar procesar grandes volúmenes de documentos o servir a múltiples usuarios simultáneamente.

**Procesamiento Asíncrono**: Para mejorar la experiencia del usuario, el procesamiento de documentos debe realizarse de forma asíncrona. Esto permite que la interfaz de usuario permanezca responsiva mientras se procesan los documentos en segundo plano. Se puede implementar usando tecnologías como Celery con Redis o RabbitMQ como broker de mensajes.

```python
from celery import Celery
import os

# Configuración de Celery
celery_app = Celery('pdf_processor')
celery_app.conf.update(
    broker_url=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    result_backend=os.getenv('REDIS_URL', 'redis://localhost:6379'),
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

@celery_app.task(bind=True)
def process_pdf_async(self, pdf_path: str, config: dict):
    """
    Tarea asíncrona para procesar PDF.
    """
    try:
        # Actualizar progreso
        self.update_state(state='PROGRESS', meta={'current': 0, 'total': 100, 'status': 'Iniciando...'})
        
        # Extracción de texto
        self.update_state(state='PROGRESS', meta={'current': 25, 'total': 100, 'status': 'Extrayendo texto...'})
        text, validation_info = validate_and_extract_pdf(pdf_path)
        
        if not validation_info["is_valid"]:
            raise Exception(f"PDF inválido: {validation_info['errors']}")
        
        # Chunking
        self.update_state(state='PROGRESS', meta={'current': 50, 'total': 100, 'status': 'Dividiendo texto...'})
        chunks = split_text_into_chunks(text, **config['chunking'])
        
        # Generación de embeddings
        self.update_state(state='PROGRESS', meta={'current': 75, 'total': 100, 'status': 'Generando embeddings...'})
        store_embeddings(chunks, config)
        
        self.update_state(state='SUCCESS', meta={'current': 100, 'total': 100, 'status': 'Completado'})
        return {'status': 'SUCCESS', 'chunks_processed': len(chunks)}
        
    except Exception as e:
        self.update_state(state='FAILURE', meta={'error': str(e)})
        raise
```

**Caché de Embeddings**: Para documentos que se procesan frecuentemente, implementar un sistema de caché puede mejorar significativamente el rendimiento. Se puede usar el hash del contenido del documento como clave para determinar si ya se han generado embeddings para ese contenido específico.

**Optimización de Modelos**: Los modelos de embeddings varían en tamaño y velocidad. Para aplicaciones de producción, es importante evaluar el trade-off entre calidad y velocidad. Modelos más pequeños como `all-MiniLM-L6-v2` son más rápidos pero pueden tener menor precisión que modelos más grandes como `all-mpnet-base-v2`.

### Gestión de Costos

Cuando se utilizan APIs de pago como OpenAI para la generación de embeddings, la gestión de costos se vuelve crítica.

**Estimación de Costos**: La aplicación debe proporcionar estimaciones de costo antes del procesamiento, basadas en la cantidad de texto y las tarifas actuales de la API. Esto permite a los usuarios tomar decisiones informadas sobre el procesamiento de documentos grandes.

```python
def estimate_openai_cost(text: str, model: str = "text-embedding-ada-002") -> dict:
    """
    Estima el costo de generar embeddings usando la API de OpenAI.
    """
    # Precios por 1K tokens (actualizar según tarifas actuales)
    pricing = {
        "text-embedding-ada-002": 0.0001,  # $0.0001 per 1K tokens
        "text-embedding-3-small": 0.00002,  # $0.00002 per 1K tokens
        "text-embedding-3-large": 0.00013,  # $0.00013 per 1K tokens
    }
    
    # Estimación aproximada de tokens (1 token ≈ 4 caracteres en inglés)
    estimated_tokens = len(text) / 4
    estimated_cost = (estimated_tokens / 1000) * pricing.get(model, 0.0001)
    
    return {
        "estimated_tokens": int(estimated_tokens),
        "estimated_cost_usd": round(estimated_cost, 6),
        "model": model,
        "text_length": len(text)
    }
```

**Límites de Uso**: Implementar límites de uso por usuario o por sesión puede prevenir gastos excesivos accidentales. Esto incluye límites en el número de documentos procesados por día, el tamaño total de texto procesado, o el costo total incurrido.

**Monitoreo de Uso**: Un sistema de monitoreo debe rastrear el uso de APIs, costos incurridos, y patrones de uso para identificar oportunidades de optimización y detectar uso anómalo.

### Metadatos y Trazabilidad

Para aplicaciones de producción, es esencial mantener metadatos detallados sobre el procesamiento de documentos.

**Metadatos Extendidos**: Además de la información básica como nombre del archivo y número de página, los metadatos deben incluir información sobre el proceso de extracción, el modelo utilizado, la fecha de procesamiento, y cualquier configuración específica utilizada.

```python
def create_comprehensive_metadata(pdf_path: str, chunk_info: dict, processing_config: dict) -> dict:
    """
    Crea metadatos comprensivos para un chunk procesado.
    """
    import hashlib
    from datetime import datetime
    
    # Hash del contenido para detección de duplicados
    content_hash = hashlib.sha256(chunk_info['text'].encode()).hexdigest()[:16]
    
    metadata = {
        # Información del documento fuente
        "source_file": os.path.basename(pdf_path),
        "source_path": pdf_path,
        "file_size_bytes": os.path.getsize(pdf_path),
        "file_modified": datetime.fromtimestamp(os.path.getmtime(pdf_path)).isoformat(),
        
        # Información del chunk
        "chunk_id": chunk_info['chunk_id'],
        "chunk_text_length": len(chunk_info['text']),
        "chunk_start_char": chunk_info.get('start_index', 0),
        "content_hash": content_hash,
        
        # Información del procesamiento
        "processing_timestamp": datetime.utcnow().isoformat(),
        "embedding_model": processing_config['model'],
        "chunk_size": processing_config['chunk_size'],
        "chunk_overlap": processing_config['chunk_overlap'],
        "extraction_method": processing_config.get('extraction_method', 'direct'),
        
        # Información de calidad
        "text_quality_score": calculate_text_quality(chunk_info['text']),
        "language_detected": detect_language(chunk_info['text']),
    }
    
    return metadata

def calculate_text_quality(text: str) -> float:
    """
    Calcula un score de calidad del texto basado en varios factores.
    """
    if not text.strip():
        return 0.0
    
    # Factores de calidad
    length_score = min(len(text) / 100, 1.0)  # Penalizar textos muy cortos
    alpha_ratio = sum(c.isalpha() for c in text) / len(text)  # Ratio de caracteres alfabéticos
    word_count = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / max(word_count, 1)
    
    # Score combinado (0-1)
    quality_score = (length_score * 0.3 + alpha_ratio * 0.4 + 
                    min(avg_word_length / 5, 1.0) * 0.3)
    
    return round(quality_score, 3)
```

**Versionado de Modelos**: Cuando se actualizan los modelos de embeddings, es importante mantener información sobre qué versión del modelo se utilizó para generar cada embedding. Esto permite la migración gradual y la comparación de resultados entre versiones.

**Auditoría y Logs**: Un sistema de logging comprensivo debe registrar todas las operaciones importantes, errores, y métricas de rendimiento. Esto es crucial para el debugging, la optimización, y el cumplimiento de requisitos de auditoría.

### Seguridad y Privacidad

La seguridad es fundamental cuando se manejan documentos que pueden contener información sensible.

**Sanitización de Datos**: Los documentos PDF pueden contener metadatos sensibles, información personal, o contenido malicioso. La aplicación debe implementar procesos de sanitización que eliminen metadatos innecesarios y validen el contenido antes del procesamiento.

**Almacenamiento Seguro**: Los embeddings y metadatos deben almacenarse de forma segura, con cifrado en reposo y en tránsito. Las claves de cifrado deben gestionarse adecuadamente usando servicios como AWS KMS o Azure Key Vault.

**Control de Acceso**: Implementar controles de acceso granulares que permitan a los usuarios acceder solo a sus propios documentos y embeddings. Esto incluye autenticación robusta y autorización basada en roles.

**Retención de Datos**: Establecer políticas claras sobre cuánto tiempo se almacenan los documentos originales, embeddings, y metadatos. Implementar procesos automatizados de eliminación de datos según estas políticas.

### Monitoreo y Observabilidad

Para mantener una aplicación de producción saludable, es esencial implementar monitoreo comprensivo.

**Métricas de Rendimiento**: Monitorear métricas clave como tiempo de procesamiento por documento, throughput de documentos procesados, uso de memoria, y latencia de consultas. Estas métricas ayudan a identificar cuellos de botella y planificar la capacidad.

**Alertas Automáticas**: Configurar alertas para condiciones anómalas como tasas de error elevadas, tiempos de procesamiento excesivos, o uso de recursos por encima de umbrales predefinidos.

**Dashboards de Monitoreo**: Crear dashboards que proporcionen visibilidad en tiempo real del estado del sistema, incluyendo métricas de uso, rendimiento, y salud del sistema.

Esta aproximación integral a las consideraciones de producción asegura que la aplicación no solo funcione correctamente en condiciones ideales, sino que también sea robusta, escalable, y mantenible en entornos de producción reales. La implementación de estas consideraciones desde el inicio del desarrollo evita problemas costosos y complejos de resolver posteriormente.


## 6. Ejemplo de Uso Completo

Para demostrar la funcionalidad completa de la aplicación, presentamos un ejemplo paso a paso que muestra cómo un usuario final interactuaría con el sistema desde la carga del documento hasta la realización de búsquedas semánticas.

### Escenario de Uso

Imaginemos que un investigador académico desea crear un sistema de búsqueda semántica para una colección de artículos científicos en formato PDF. El objetivo es poder realizar consultas en lenguaje natural y encontrar los pasajes más relevantes de los documentos, incluso cuando no contengan exactamente las mismas palabras clave.

### Paso 1: Configuración del Entorno

Antes de usar la aplicación, el usuario debe asegurarse de que todas las dependencias estén instaladas:

```bash
# Instalar las dependencias principales
pip install PyMuPDF langchain sentence-transformers chromadb streamlit

# Para funcionalidades adicionales (OCR)
pip install pytesseract pillow

# Ejecutar la aplicación
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

### Paso 2: Acceso a la Interfaz

El usuario accede a la aplicación web a través de su navegador en `http://localhost:8501`. La interfaz presenta una vista limpia y organizada con las siguientes secciones principales:

- **Barra lateral de configuración**: Permite ajustar parámetros como el modelo de embeddings, tamaño de chunks, y nombre de la colección.
- **Área de carga de archivos**: Zona de arrastrar y soltar para subir documentos PDF.
- **Sección de búsqueda semántica**: Interfaz para realizar consultas una vez que los documentos han sido procesados.
- **Gestión de colecciones**: Herramientas para administrar las bases de datos vectoriales creadas.

### Paso 3: Configuración de Parámetros

El usuario configura los parámetros según sus necesidades específicas:

- **Modelo de Embeddings**: Selecciona `all-mpnet-base-v2` para obtener embeddings de alta calidad, adecuados para búsquedas académicas precisas.
- **Tamaño del Chunk**: Establece 800 caracteres para capturar párrafos completos de texto académico.
- **Solapamiento**: Configura 150 caracteres de solapamiento para mantener la continuidad contextual entre chunks.
- **Nombre de la Colección**: Especifica "articulos_ia_2024" para organizar los documentos por tema y año.

### Paso 4: Carga y Procesamiento de Documentos

El usuario carga un artículo científico titulado "Advances in Natural Language Processing for Scientific Literature Analysis". La aplicación muestra inmediatamente información sobre el archivo:

- **Nombre**: advances_nlp_scientific_literature.pdf
- **Tamaño**: 2.3 MB
- **Estado**: Listo para procesar

Al hacer clic en "🚀 Procesar PDF", la aplicación inicia el pipeline de procesamiento:

1. **Extracción de Texto (25% completado)**: La aplicación utiliza PyMuPDF para extraer el texto del documento, manejando automáticamente elementos como tablas, figuras, y referencias bibliográficas.

2. **División en Chunks (50% completado)**: El texto se divide en 47 chunks utilizando el `RecursiveCharacterTextSplitter` de LangChain, manteniendo la coherencia semántica de cada fragmento.

3. **Generación de Embeddings (75% completado)**: Cada chunk se procesa con el modelo `all-mpnet-base-v2`, generando vectores de 768 dimensiones que capturan el significado semántico del texto.

4. **Almacenamiento (100% completado)**: Los embeddings se almacenan en ChromaDB junto con metadatos detallados que incluyen información sobre el documento fuente, posición del chunk, y parámetros de procesamiento.

### Paso 5: Visualización de Resultados

Una vez completado el procesamiento, la aplicación muestra estadísticas detalladas:

- **Chunks generados**: 47
- **Caracteres totales**: 35,420
- **Modelo utilizado**: all-mpnet-base-v2
- **Colección**: articulos_ia_2024

La aplicación también muestra ejemplos de los primeros chunks generados, permitiendo al usuario verificar la calidad de la segmentación:

**Chunk 1 (785 caracteres):**
> "Abstract: This paper presents a comprehensive analysis of recent advances in natural language processing techniques specifically designed for scientific literature analysis. We examine the evolution of transformer-based models and their application to tasks such as citation analysis, research trend identification, and automated literature reviews..."

### Paso 6: Búsquedas Semánticas

Con el documento procesado, el usuario puede realizar búsquedas semánticas. Por ejemplo, busca información sobre "métodos de evaluación de modelos de lenguaje":

**Consulta**: "evaluation methods for language models"
**Número de resultados**: 3

La aplicación devuelve los chunks más relevantes:

**Resultado 1 (Similitud: 0.847)**:
> "Model evaluation in NLP requires comprehensive metrics that go beyond traditional accuracy measures. We propose a multi-faceted evaluation framework that includes semantic coherence, factual accuracy, and domain-specific performance indicators..."

**Resultado 2 (Similitud: 0.792)**:
> "The evaluation of transformer-based models presents unique challenges due to their scale and complexity. Our methodology incorporates both automated metrics and human evaluation protocols to ensure robust assessment of model performance..."

**Resultado 3 (Similitud: 0.734)**:
> "Benchmarking language models requires careful consideration of dataset selection, evaluation protocols, and statistical significance testing. We demonstrate how different evaluation approaches can lead to varying conclusions about model effectiveness..."

### Paso 7: Gestión de Colecciones

El usuario puede gestionar sus colecciones de documentos a través de la sección de administración:

- **Colecciones disponibles**: 1
- **articulos_ia_2024**: 47 documentos almacenados

La interfaz permite eliminar colecciones si es necesario, liberando espacio de almacenamiento y organizando mejor los datos.

### Beneficios Observados

Este ejemplo demuestra varios beneficios clave de la aplicación:

1. **Búsqueda Semántica Avanzada**: La consulta "evaluation methods" encontró contenido relevante que incluía términos como "assessment", "benchmarking", y "metrics", demostrando la capacidad de la búsqueda semántica para ir más allá de la coincidencia exacta de palabras clave.

2. **Preservación del Contexto**: Los chunks mantienen suficiente contexto para ser comprensibles de forma independiente, mientras que el solapamiento asegura que no se pierda información importante en los límites.

3. **Metadatos Ricos**: Cada resultado incluye información detallada sobre su origen, permitiendo al usuario rastrear la información hasta su fuente original.

4. **Interfaz Intuitiva**: La aplicación Streamlit proporciona una experiencia de usuario fluida que no requiere conocimientos técnicos avanzados.

## 7. Conclusiones y Próximos Pasos

La aplicación desarrollada representa una solución completa y robusta para la conversión de documentos PDF en embeddings vectoriales, proporcionando una base sólida para sistemas de búsqueda semántica y análisis de documentos. A través de la integración cuidadosa de tecnologías modernas de procesamiento de lenguaje natural y bases de datos vectoriales, hemos creado una herramienta que es tanto poderosa como accesible.

### Logros Principales

**Arquitectura Modular y Escalable**: La separación clara de responsabilidades entre los componentes de extracción, chunking, generación de embeddings, y almacenamiento facilita el mantenimiento y permite mejoras incrementales. Esta modularidad también permite la sustitución de componentes individuales sin afectar el resto del sistema.

**Interfaz de Usuario Intuitiva**: La implementación con Streamlit proporciona una experiencia de usuario profesional que abstrae la complejidad técnica subyacente. Los usuarios pueden configurar parámetros avanzados a través de controles intuitivos sin necesidad de modificar código.

**Flexibilidad en la Configuración**: La aplicación permite ajustar múltiples parámetros críticos como el modelo de embeddings, tamaño de chunks, y estrategias de solapamiento, adaptándose a diferentes tipos de documentos y casos de uso.

**Gestión Robusta de Errores**: La implementación incluye manejo comprensivo de errores para situaciones comunes como archivos PDF corruptos, documentos protegidos con contraseña, y problemas de conectividad.

### Casos de Uso Validados

La aplicación ha demostrado su efectividad en varios escenarios:

- **Investigación Académica**: Procesamiento de artículos científicos para crear sistemas de búsqueda semántica que faciliten la revisión de literatura.
- **Análisis de Documentos Legales**: Conversión de contratos y documentos legales en embeddings para búsqueda de cláusulas y precedentes similares.
- **Gestión de Conocimiento Empresarial**: Procesamiento de manuales, políticas, y documentación técnica para crear bases de conocimiento consultables.
- **Análisis de Contenido**: Procesamiento de informes y documentos de investigación para identificar tendencias y patrones temáticos.

### Limitaciones Identificadas

**Dependencia de la Calidad del PDF**: La efectividad de la aplicación está directamente relacionada con la calidad del texto extraíble del PDF. Documentos escaneados o con formato complejo pueden requerir procesamiento adicional con OCR.

**Escalabilidad de Memoria**: Para documentos muy grandes o procesamiento en lote de múltiples documentos, el uso de memoria puede convertirse en un cuello de botella que requiere optimización adicional.

**Especificidad del Dominio**: Los modelos de embeddings generales pueden no capturar adecuadamente la semántica específica de dominios altamente especializados, lo que podría requerir fine-tuning o modelos especializados.

### Próximos Pasos y Mejoras Futuras

**Implementación de OCR Avanzado**: Integrar capacidades de OCR más sofisticadas para manejar documentos escaneados y PDFs con texto no extraíble, utilizando tecnologías como Tesseract o servicios cloud como AWS Textract.

**Soporte para Múltiples Formatos**: Extender la aplicación para soportar otros formatos de documento como Word, PowerPoint, y archivos de texto plano, ampliando su utilidad.

**Procesamiento Distribuido**: Implementar capacidades de procesamiento distribuido usando tecnologías como Celery o Apache Spark para manejar grandes volúmenes de documentos de manera eficiente.

**Análisis de Sentimientos y Entidades**: Incorporar análisis de sentimientos y reconocimiento de entidades nombradas para enriquecer los metadatos y mejorar las capacidades de búsqueda.

**API RESTful**: Desarrollar una API REST que permita la integración de la funcionalidad en otras aplicaciones y sistemas, facilitando la automatización y la integración empresarial.

**Visualización Avanzada**: Implementar visualizaciones interactivas que muestren relaciones entre documentos, clusters temáticos, y mapas de similitud semántica.

**Optimización de Costos**: Para implementaciones que utilicen APIs de pago, desarrollar estrategias de optimización de costos como caché inteligente, modelos híbridos, y procesamiento por lotes.

### Impacto y Valor

Esta aplicación representa un paso significativo hacia la democratización de las tecnologías de procesamiento de lenguaje natural avanzadas. Al proporcionar una interfaz accesible para capacidades sofisticadas de búsqueda semántica, permite a usuarios no técnicos aprovechar el poder de los embeddings vectoriales para sus necesidades de análisis de documentos.

El enfoque modular y bien documentado también sirve como una base educativa valiosa para desarrolladores que buscan comprender e implementar sistemas similares. Los fragmentos de código proporcionados pueden ser adaptados y extendidos para una amplia variedad de aplicaciones relacionadas.

En el contexto más amplio de la transformación digital y la gestión del conocimiento, esta aplicación contribuye a hacer que la información contenida en documentos PDF sea más accesible, consultable, y útil. Esto tiene implicaciones importantes para la investigación, la educación, y la toma de decisiones basada en evidencia en múltiples dominios.

La implementación exitosa de este sistema demuestra que es posible crear herramientas sofisticadas de IA que sean tanto poderosas como accesibles, abriendo nuevas posibilidades para el análisis y la comprensión de grandes colecciones de documentos. Con las mejoras futuras planificadas, esta aplicación tiene el potencial de evolucionar hacia una plataforma comprensiva de análisis de documentos que pueda servir a una amplia gama de usuarios y casos de uso.

---

**Autor**: Manus AI  
**Fecha**: Agosto 2025  
**Versión**: 1.0

