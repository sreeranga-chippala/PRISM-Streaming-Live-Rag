from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator
import csv
import json


# ============================================================
# COMMON DOCUMENT REPRESENTATION
# ============================================================

@dataclass
class Document:
    """
    Standard representation used by the rest of the RAG pipeline.

    Every supported file format is converted into this structure.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================
# DOCUMENT LOADER
# ============================================================

class DocumentLoader:

    # --------------------------------------------------------
    # SUPPORTED FILE FORMATS
    # --------------------------------------------------------

    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",

        # Programming languages
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".kts",

        # Web / markup
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".xml",

        # Data / configuration
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".conf",

        # Query / shell
        ".sql",
        ".sh",
        ".bat",
        ".ps1",
    }

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif",
    }

    # --------------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------------

    def load(
        self,
        file_path: str | Path,
    ) -> list[Document]:
        """
        Convenient API.

        Loads a file and returns all extracted documents
        as a list.

        Suitable for small/normal files and testing.

        For large files use iter_load().
        """

        return list(
            self.iter_load(file_path)
        )

    def iter_load(
        self,
        file_path: str | Path,
    ) -> Iterator[Document]:
        """
        Memory-friendly API.

        Yields documents incrementally instead of returning
        the complete list at once.

        This is the API that our large-scale ingestion
        pipeline should use.
        """

        file_path = Path(file_path)

        self._validate_file(file_path)

        extension = file_path.suffix.lower()

        try:

            # ------------------------------------------------
            # PDF
            # ------------------------------------------------

            if extension == ".pdf":

                yield from self._iter_load_pdf(
                    file_path
                )

            # ------------------------------------------------
            # DOCX
            # ------------------------------------------------

            elif extension == ".docx":

                yield from self._iter_load_docx(
                    file_path
                )

            # ------------------------------------------------
            # PPTX
            # ------------------------------------------------

            elif extension == ".pptx":

                yield from self._iter_load_pptx(
                    file_path
                )

            # ------------------------------------------------
            # XLSX / XLS
            # ------------------------------------------------

            elif extension in {".xlsx", ".xls"}:

                yield from self._iter_load_excel(
                    file_path
                )

            # ------------------------------------------------
            # CSV
            # ------------------------------------------------

            elif extension == ".csv":

                yield from self._iter_load_csv(
                    file_path
                )

            # ------------------------------------------------
            # JSON
            # ------------------------------------------------

            elif extension == ".json":

                yield from self._iter_load_json(
                    file_path
                )

            # ------------------------------------------------
            # TEXT / CODE / MARKUP
            # ------------------------------------------------

            elif extension in self.TEXT_EXTENSIONS:

                yield from self._iter_load_text(
                    file_path
                )

            # ------------------------------------------------
            # IMAGE
            # ------------------------------------------------

            elif extension in self.IMAGE_EXTENSIONS:

                yield from self._iter_load_image(
                    file_path
                )

            # ------------------------------------------------
            # UNSUPPORTED
            # ------------------------------------------------

            else:

                raise ValueError(
                    f"Unsupported file format: "
                    f"{extension or '[no extension]'}"
                )

        except Exception as error:

            raise RuntimeError(
                f"Failed to load file "
                f"'{file_path}': {error}"
            ) from error

    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_file(
        self,
        file_path: Path,
    ) -> None:

        if not file_path.exists():

            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        if not file_path.is_file():

            raise ValueError(
                f"Path is not a file: {file_path}"
            )

        if file_path.stat().st_size == 0:

            raise ValueError(
                f"File is empty: {file_path}"
            )

    # ========================================================
    # COMMON METADATA
    # ========================================================

    def _base_metadata(
        self,
        file_path: Path,
        file_type: str | None = None,
    ) -> dict[str, Any]:

        return {
            "source": file_path.name,
            "file_path": str(file_path),
            "file_type": (
                file_type
                if file_type
                else file_path.suffix.lower()
            ),
            "file_size_bytes": file_path.stat().st_size,
        }

    # ========================================================
    # TEXT / CODE / MARKUP
    # ========================================================

    def _iter_load_text(
        self,
        file_path: Path,
    ) -> Iterator[Document]:
        """
        Read text incrementally.

        Instead of read_text(), we process the file line by line.
        """

        metadata = self._base_metadata(
            file_path
        )

        buffer = []

        buffer_size = 100

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for line in file:

                buffer.append(
                    line.rstrip("\n")
                )

                if len(buffer) >= buffer_size:

                    text = "\n".join(buffer).strip()

                    if text:

                        yield Document(
                            text=text,
                            metadata=metadata.copy(),
                        )

                    buffer.clear()

        # Process remaining lines

        if buffer:

            text = "\n".join(buffer).strip()

            if text:

                yield Document(
                    text=text,
                    metadata=metadata.copy(),
                )

    # ========================================================
    # PDF
    # ========================================================

    def _iter_load_pdf(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        from pypdf import PdfReader

        reader = PdfReader(
            str(file_path)
        )

        total_pages = len(
            reader.pages
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            try:

                text = page.extract_text() or ""

            except Exception as error:

                print(
                    f"[WARNING] Could not extract "
                    f"page {page_number} "
                    f"from {file_path.name}: {error}"
                )

                continue

            text = text.strip()

            if not text:

                continue

            metadata = self._base_metadata(
                file_path,
                ".pdf",
            )

            metadata.update({
                "page": page_number,
                "total_pages": total_pages,
            })

            yield Document(
                text=text,
                metadata=metadata,
            )

    # ========================================================
    # DOCX
    # ========================================================

    def _iter_load_docx(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        from docx import Document as DocxDocument

        doc = DocxDocument(
            str(file_path)
        )

        metadata = self._base_metadata(
            file_path,
            ".docx",
        )

        paragraph_buffer = []

        for paragraph in doc.paragraphs:

            text = paragraph.text.strip()

            if not text:

                continue

            paragraph_buffer.append(text)

            # Process blocks periodically

            if len(paragraph_buffer) >= 50:

                yield Document(
                    text="\n".join(
                        paragraph_buffer
                    ),
                    metadata=metadata.copy(),
                )

                paragraph_buffer.clear()

        # Remaining paragraphs

        if paragraph_buffer:

            yield Document(
                text="\n".join(
                    paragraph_buffer
                ),
                metadata=metadata.copy(),
            )

    # ========================================================
    # PPTX
    # ========================================================

    def _iter_load_pptx(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        from pptx import Presentation

        presentation = Presentation(
            str(file_path)
        )

        total_slides = len(
            presentation.slides
        )

        for slide_number, slide in enumerate(
            presentation.slides,
            start=1,
        ):

            texts = []

            for shape in slide.shapes:

                if not hasattr(
                    shape,
                    "text",
                ):
                    continue

                text = shape.text.strip()

                if text:

                    texts.append(text)

            slide_text = "\n".join(
                texts
            ).strip()

            if not slide_text:

                continue

            metadata = self._base_metadata(
                file_path,
                ".pptx",
            )

            metadata.update({
                "slide": slide_number,
                "total_slides": total_slides,
            })

            yield Document(
                text=slide_text,
                metadata=metadata,
            )

    # ========================================================
    # CSV
    # ========================================================

    def _iter_load_csv(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        metadata = self._base_metadata(
            file_path,
            ".csv",
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace",
            newline="",
        ) as file:

            reader = csv.DictReader(
                file
            )

            row_number = 0

            for row in reader:

                row_number += 1

                text = json.dumps(
                    row,
                    ensure_ascii=False,
                )

                row_metadata = metadata.copy()

                row_metadata["row"] = row_number

                yield Document(
                    text=text,
                    metadata=row_metadata,
                )

    # ========================================================
    # JSON
    # ========================================================

    def _iter_load_json(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        metadata = self._base_metadata(
            file_path,
            ".json",
        )

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            data = json.load(file)

        # JSON itself may be very large, so we split
        # large arrays/objects into individual records.

        if isinstance(data, list):

            for index, item in enumerate(
                data
            ):

                text = json.dumps(
                    item,
                    ensure_ascii=False,
                )

                item_metadata = metadata.copy()

                item_metadata["record"] = index

                yield Document(
                    text=text,
                    metadata=item_metadata,
                )

        else:

            text = json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
            )

            yield Document(
                text=text,
                metadata=metadata,
            )

    # ========================================================
    # EXCEL
    # ========================================================

    def _iter_load_excel(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        import pandas as pd

        excel_file = pd.ExcelFile(
            file_path
        )

        for sheet_name in excel_file.sheet_names:

            try:

                dataframe = pd.read_excel(
                    file_path,
                    sheet_name=sheet_name,
                )

            except Exception as error:

                print(
                    f"[WARNING] Could not read "
                    f"sheet '{sheet_name}': {error}"
                )

                continue

            metadata = self._base_metadata(
                file_path,
                ".xlsx",
            )

            metadata["sheet"] = sheet_name

            # Process rows individually rather than
            # converting the entire sheet into one
            # enormous string.

            for row_number, row in dataframe.iterrows():

                row_data = row.to_dict()

                text = json.dumps(
                    row_data,
                    ensure_ascii=False,
                    default=str,
                )

                row_metadata = metadata.copy()

                row_metadata["row"] = (
                    int(row_number)
                )

                yield Document(
                    text=text,
                    metadata=row_metadata,
                )

    # ========================================================
    # IMAGE / OCR
    # ========================================================

    def _iter_load_image(
        self,
        file_path: Path,
    ) -> Iterator[Document]:

        from PIL import Image
        import pytesseract

        metadata = self._base_metadata(
            file_path,
            "image",
        )

        try:

            with Image.open(
                file_path
            ) as image:

                text = pytesseract.image_to_string(
                    image
                )

        except Exception as error:

            raise RuntimeError(
                f"OCR failed for "
                f"'{file_path}': {error}"
            ) from error

        text = text.strip()

        if text:

            yield Document(
                text=text,
                metadata=metadata,
            )