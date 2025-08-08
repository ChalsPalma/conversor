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


