# Conversor de PDF a Embeddings Vectoriales

Una aplicación completa para convertir documentos PDF en embeddings vectoriales, permitiendo búsquedas semánticas avanzadas y análisis de documentos.

## 🚀 Características Principales

- **Extracción de Texto Robusta**: Utiliza PyMuPDF para extraer texto de PDFs con manejo de errores avanzado
- **Chunking Inteligente**: División de texto usando LangChain con solapamiento configurable
- **Embeddings de Alta Calidad**: Soporte para múltiples modelos de Sentence-Transformers
- **Base de Datos Vectorial**: Almacenamiento eficiente con ChromaDB
- **Interfaz Web Intuitiva**: Aplicación Streamlit fácil de usar
- **Búsqueda Semántica**: Consultas en lenguaje natural con resultados relevantes
- **Gestión de Colecciones**: Organización y administración de documentos procesados

## 📋 Requisitos del Sistema

- Python 3.8 o superior
- 4GB de RAM mínimo (8GB recomendado)
- 2GB de espacio libre en disco
- Conexión a internet para descargar modelos

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone <repository-url>
cd pdf-to-embeddings
```

2. **Crear entorno virtual**:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**:
```bash
pip install -r requirements.txt
```

4. **Verificar instalación**:
```bash
python pdf_extractor.py
python text_chunker.py
python embedding_processor.py
```

## 🚀 Uso Rápido

### Ejecutar la Aplicación Web

```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

Accede a la aplicación en: `http://localhost:8501`

### Uso Programático

```python
from pdf_extractor import extract_text_from_pdf
from text_chunker import split_text_into_chunks
from embedding_processor import generate_embeddings_and_store

# Procesar un PDF
text = extract_text_from_pdf("documento.pdf")
chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=100)
generate_embeddings_and_store("documento.pdf", collection_name="mi_coleccion")
```

## 📁 Estructura del Proyecto

```
pdf-to-embeddings/
├── pdf_extractor.py          # Extracción de texto de PDFs
├── text_chunker.py           # División de texto en chunks
├── embedding_processor.py    # Generación y almacenamiento de embeddings
├── streamlit_app.py          # Interfaz web principal
├── requirements.txt          # Dependencias del proyecto
├── README.md                # Este archivo
└── chroma_db/               # Base de datos vectorial (se crea automáticamente)
```

## ⚙️ Configuración

### Modelos de Embeddings Disponibles

- `all-MiniLM-L6-v2`: Rápido y eficiente (384 dimensiones)
- `all-mpnet-base-v2`: Alta calidad (768 dimensiones)
- `paraphrase-MiniLM-L6-v2`: Optimizado para paráfrasis
- `distilbert-base-nli-stsb-mean-tokens`: Basado en BERT

### Parámetros de Chunking

- **Tamaño del Chunk**: 100-2000 caracteres (recomendado: 500)
- **Solapamiento**: 0-500 caracteres (recomendado: 100)

## 🔍 Ejemplos de Uso

### Búsqueda Semántica

```python
import chromadb
from chromadb.utils import embedding_functions

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("mi_coleccion")

results = collection.query(
    query_texts=["inteligencia artificial"],
    n_results=3
)

for doc, distance in zip(results['documents'][0], results['distances'][0]):
    print(f"Similitud: {1-distance:.3f}")
    print(f"Texto: {doc}\n")
```

### Procesamiento por Lotes

```python
import os
from pathlib import Path

pdf_directory = "documentos/"
for pdf_file in Path(pdf_directory).glob("*.pdf"):
    print(f"Procesando: {pdf_file}")
    generate_embeddings_and_store(
        str(pdf_file), 
        collection_name="documentos_lote"
    )
```

## 🐛 Solución de Problemas

### Error: "No se pudo extraer texto del PDF"
- Verifica que el PDF no esté protegido con contraseña
- Asegúrate de que el archivo no esté corrupto
- Para PDFs escaneados, considera usar OCR

### Error de memoria con documentos grandes
- Reduce el tamaño del chunk
- Procesa el documento en secciones
- Aumenta la memoria disponible del sistema

### Modelos no se descargan
- Verifica la conexión a internet
- Comprueba el espacio disponible en disco
- Reinicia la aplicación

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [PyMuPDF](https://pymupdf.readthedocs.io/) por la extracción de texto de PDFs
- [LangChain](https://python.langchain.com/) por las herramientas de procesamiento de texto
- [Sentence-Transformers](https://www.sbert.net/) por los modelos de embeddings
- [ChromaDB](https://www.trychroma.com/) por la base de datos vectorial
- [Streamlit](https://streamlit.io/) por el framework de interfaz web

## 📞 Soporte

Para preguntas, problemas o sugerencias:
- Abre un issue en GitHub
- Consulta la documentación completa en `plan_de_aplicacion.md`

---

**Desarrollado por**: Manus AI  
**Versión**: 1.0  
**Última actualización**: Agosto 2025

