from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_docling.loader import DoclingLoader

pipeline_options = PdfPipelineOptions()
pipeline_options.allow_external_plugins = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


def load_document(file):
    print(f"Loading document: {file}")
    try:
        loader = DoclingLoader(file, converter=converter)
        data = loader.load()
        
        print(f"Step 1 - Loaded Docs: {len(data)}")
        if not data:
            raise ValueError(
                f"No documents were found/loaded from path: {file}. "
                f"Please check file permissions and file paths."
            )

        for doc in data:
            raw_page_no = None

            # Check all common keys where page numbers might reside
            if "page" in doc.metadata:
                raw_page_no = doc.metadata["page"]
            elif "page_number" in doc.metadata:
                raw_page_no = doc.metadata["page_number"]
            elif "dl_meta" in doc.metadata and isinstance(doc.metadata["dl_meta"], dict):
                # Docling's internal metadata fallback
                raw_page_no = doc.metadata["dl_meta"].get("page")

            # Clean and filter non-serializable objects from metadata
            clean_meta = {
                k: v
                for k, v in doc.metadata.items()
                if isinstance(v, (str, int, float, bool))
                and k not in ["dl_meta", "doc_items"]
            }

            # Safely cast page number to int if possible
            parsed_page_no = None
            if raw_page_no is not None:
                try:
                    parsed_page_no = int(raw_page_no)
                    clean_meta["page"] = parsed_page_no
                except (ValueError, TypeError):
                    parsed_page_no = None

            # Safe comparison with NoneType guard
            if parsed_page_no is not None and parsed_page_no <= 14:
                clean_meta["is_index"] = True
            else:
                clean_meta["is_index"] = False

            doc.metadata = clean_meta

        return data

    except Exception as e:
        print(f"Error loading document: {e}")
        # Re-raise or return empty list depending on app requirements
        raise e