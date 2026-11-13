from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from thermodynamics.parsers.base import PDFExtractionBackend
from thermodynamics.parsers.document_model import (
    DocumentExtraction,
    DocumentMetadata,
    Table,
)
from thermodynamics.parsers.grobid_backend import GrobidBackend
from thermodynamics.parsers.pdf_processing import PDFProcessingPipeline


@dataclass(frozen=True)
class VLEExtractionResult:
    source_path: str
    metadata: DocumentMetadata
    candidate_tables: list[Table]
    extraction: DocumentExtraction

    @property
    def failures(self) -> dict[str, str]:
        return self.extraction.failures


class KeywordVLETableSelector:
    """Cheap keyword pre-filter over titles and header rows."""

    DEFAULT_KEYWORDS = (
        'vle',
        'vapor',
        'vapour',
        'liquid',
        'equilibrium',
        'x1',
        'y1',
        'mole fraction',
    )

    def __init__(self, keywords: Sequence[str] | None = None) -> None:
        self.keywords = tuple(keywords or self.DEFAULT_KEYWORDS)

    def is_candidate(self, table: Table) -> bool:
        haystack = ' '.join(
            [table.title, *(cell.content for cell in table.header_row.cells)]
        ).lower()

        return any(keyword in haystack for keyword in self.keywords)

    def select(self, tables: Sequence[Table]) -> list[Table]:
        return [table for table in tables if self.is_candidate(table)]


class VLEDataExtractor:
    """Domain layer: picks VLE tables out of whatever the backends produced."""

    def __init__(
        self,
        *,
        backends: Sequence[PDFExtractionBackend] | None = None,
        table_selector: KeywordVLETableSelector | None = None,
        fail_fast: bool = False,
        verbose: bool = True,
    ) -> None:

        self.pipeline = PDFProcessingPipeline(
            backends=backends or [GrobidBackend(verbose=verbose)],
            fail_fast=fail_fast,
            verbose=verbose,
        )
        self.table_selector = table_selector or KeywordVLETableSelector()

    def extract(self, pdf_path: str | Path) -> VLEExtractionResult:

        extraction = self.pipeline.process(pdf_path)

        return VLEExtractionResult(
            source_path=extraction.source_path,
            metadata=extraction.metadata,
            candidate_tables=self.table_selector.select(extraction.tables),
            extraction=extraction,
        )

    def close(self) -> None:
        self.pipeline.close()
