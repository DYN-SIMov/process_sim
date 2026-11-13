from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from src.thermodynamics.parsers.document_model import ParsedDocument


class BackendError(RuntimeError):
    """Raised when a PDF extraction backend fails to process a document."""


@runtime_checkable
class PDFExtractionBackend(Protocol):
    """Anything that can turn a PDF into the neutral document model.

    Implementations are free to call a local library, an HTTP service, a
    containerised tool or an LLM: the pipeline only depends on this contract.
    """

    @property
    def name(self) -> str: ...

    def extract(self, pdf_path: str | Path) -> ParsedDocument: ...

    def close(self) -> None: ...


@runtime_checkable
class RawPayloadParser(Protocol):
    """Converts the raw output of one backend into the neutral document model.

    Keeping this separate from the backend lets the transport (HTTP, subprocess)
    and the format-specific parsing (TEI XML, JSON, ...) vary independently.
    """

    def parse(
        self,
        raw_payload: str,
        *,
        source_path: str,
        backend_name: str,
    ) -> ParsedDocument: ...
