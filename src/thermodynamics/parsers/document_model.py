from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableCell:
    content: str
    coords: str = ''


@dataclass(frozen=True)
class TableRow:
    cells: list[TableCell]
    coords: str = ''


@dataclass(frozen=True)
class Table:
    content_rows: list[TableRow]
    header_row: TableRow
    title: str = ''
    coords: str = ''
    page: int | None = None
    source_backend: str = ''


@dataclass(frozen=True)
class DocumentMetadata:
    authors: list[str] = field(default_factory=list)
    publication_date: str = ''
    title: str = ''
    journal_name: str = ''
    doi: str = ''


MERGEABLE_METADATA_FIELDS = (
    'authors',
    'publication_date',
    'title',
    'journal_name',
    'doi',
)


@dataclass(frozen=True)
class ParsedDocument:
    """What a single backend managed to extract from a single PDF."""

    source_path: str
    backend_name: str
    metadata: DocumentMetadata
    tables: list[Table]
    # Untouched backend output (TEI XML, JSON, plain text, ...) kept for debugging
    # and for re-parsing without re-running the backend.
    raw_payload: str | None = None


@dataclass(frozen=True)
class DocumentExtraction:
    """Aggregated result of running one or several backends on the same PDF."""

    source_path: str
    documents: list[ParsedDocument]
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def tables(self) -> list[Table]:
        return [table for document in self.documents for table in document.tables]

    @property
    def metadata(self) -> DocumentMetadata:
        """First non-empty value per field, following backend priority order."""

        merged: dict[str, object] = {}
        for document in self.documents:
            for field_name in MERGEABLE_METADATA_FIELDS:
                if merged.get(field_name):
                    continue
                value = getattr(document.metadata, field_name)
                if value:
                    merged[field_name] = value

        return DocumentMetadata(**merged)  # type: ignore[arg-type]

    def document_by_backend(self, backend_name: str) -> ParsedDocument | None:
        for document in self.documents:
            if document.backend_name == backend_name:
                return document
        return None
