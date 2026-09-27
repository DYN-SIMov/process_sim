from dataclasses import dataclass

import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class TableCell: 
    content: str | float | int
    coords: str


@dataclass(frozen=True)
class TableRow:
    cells: list[TableCell]
    coords: str


@dataclass(frozen=True)
class Table:
    content_rows: list[TableRow]
    header_row: TableRow
    title: str
    coords: str



class XMLParser:

    def __init__(self, xml_file):
        self.xml_file = xml_file
        self.root = ET.fromstring(xml_file)


    def _local_name(self, tag):
        """Extract the local name from an XML tag."""
        if '}' in tag:
            return tag.split('}', 1)[1]
        return tag

    def _element_text(self, element):
        """Extract the text content from an XML element."""
        return element.text.strip() if element.text else ''

    def _normalize_text(self, text):
        """Normalize text by removing extra whitespace and newlines."""
        return ' '.join(text.split())

    def _find_first_by_local_name(self, element, local_name):
        """Find the first descendant element (including self) with the given local tag name."""
        for descendant in element.iter():
            if self._local_name(descendant.tag) == local_name:
                return descendant
        return None

    def _find_all_by_local_name(self, element, local_name):
        """Find all descendant elements (including self) with the given local tag name."""
        return [d for d in element.iter() if self._local_name(d.tag) == local_name]

    def _full_text(self, element):
        """Extract the full text content of an element, including text of child elements."""
        return self._normalize_text(''.join(element.itertext()))

    def _get_author_names(self, element) -> list[str]:

        # Restrict the search to the teiHeader: cited references in
        # text/back/listBibl also contain author elements that must be excluded.
        tei_header = self._find_first_by_local_name(element, 'teiHeader')
        search_scope = tei_header if tei_header is not None else element

        # Within the header, the article's own authors are listed under
        # sourceDesc/biblStruct/analytic (monogr holds journal-level info).
        analytic = self._find_first_by_local_name(search_scope, 'analytic')
        if analytic is not None:
            search_scope = analytic

        authors = []
        for author in self._find_all_by_local_name(search_scope, 'author'):
            pers_name = self._find_first_by_local_name(author, 'persName')
            if pers_name is None:
                continue

            forenames = [
                self._element_text(forename)
                for forename in self._find_all_by_local_name(pers_name, 'forename')
            ]
            surname_element = self._find_first_by_local_name(pers_name, 'surname')
            surname = self._element_text(surname_element) if surname_element is not None else ''

            full_name = self._normalize_text(' '.join([*forenames, surname]))
            if full_name:
                authors.append(full_name)

        return authors

    def _get_publication_date(self, element) -> str:

        imprint = self._find_first_by_local_name(element, 'imprint')
        if imprint is None:
            return ''

        dates = self._find_all_by_local_name(imprint, 'date')
        published_dates = [d for d in dates if d.get('type') == 'published']

        for date in published_dates or dates:
            when = date.get('when')
            if when:
                return when.strip()
            text = self._full_text(date)
            if text:
                return text

        return ''

    def _get_title(self, element) -> str:

        title_stmt = self._find_first_by_local_name(element, 'titleStmt')
        if title_stmt is None:
            return ''

        titles = self._find_all_by_local_name(title_stmt, 'title')
        main_titles = [t for t in titles if t.get('type') == 'main']

        for title in main_titles or titles:
            text = self._full_text(title)
            if text:
                return text

        return ''

    def _get_journal_name(self, element) -> str:

        monogr = self._find_first_by_local_name(element, 'monogr')
        if monogr is None:
            return ''

        titles = self._find_all_by_local_name(monogr, 'title')
        journal_titles = [t for t in titles if t.get('level') in ('j', 'm')]

        for title in journal_titles or titles:
            text = self._full_text(title)
            if text:
                return text

        return ''


    def get_metadata(self):

        authors = self._get_author_names(self.root)
        publication_date = self._get_publication_date(self.root)
        title = self._get_title(self.root)
        journal_name = self._get_journal_name(self.root)

        metadata = {
            "authors": authors,
            "publication_date": publication_date,
            "title": title, 
            "journal_name": journal_name
        }

        return metadata


    def _is_numeric_cell(self, content: str) -> bool:
        """Check whether a cell content can be read as a single number."""
        candidate = content.replace('\u2212', '-').replace(',', '.').strip()
        try:
            float(candidate)
        except ValueError:
            return False
        return True


    def _build_table_row(self, row_element) -> TableRow:

        cells = [
            TableCell(
                content=self._full_text(cell_element),
                coords=cell_element.get('coords', ''),
            )
            for cell_element in self._find_all_by_local_name(row_element, 'cell')
        ]

        return TableRow(cells=cells, coords=row_element.get('coords', ''))


    def _get_table_title(self, figure_element) -> str:
        """Concatenate the table label/head and its caption into a single title."""

        title_parts = []
        for local_name in ('head', 'figDesc'):
            part_element = self._find_first_by_local_name(figure_element, local_name)
            if part_element is not None:
                text = self._full_text(part_element)
                if text:
                    title_parts.append(text)

        return self._normalize_text(' '.join(title_parts))


    def _split_header_and_content_rows(self, rows: list[TableRow]) -> tuple[TableRow, list[TableRow]]:
        """Treat the leading non-numeric rows as the header; the rest is presumed to be data.
        NOTE: relying on this assumption may not always be correct; implementation needs to be robustified later """

        empty_row = TableRow(cells=[], coords='')
        if not rows:
            return empty_row, []

        for index, row in enumerate(rows):
            if row.cells and all(self._is_numeric_cell(cell.content) for cell in row.cells):
                if index == 0:
                    return empty_row, rows
                return rows[index - 1], rows[index:]

        return rows[0], rows[1:]


    def _is_table_figure(self, figure_element) -> bool:
        """GROBID tags table figures with type="table"; plain graphic figures carry no type."""

        if figure_element.get('type') == 'table':
            return True
        return self._find_first_by_local_name(figure_element, 'table') is not None


    def _detect_tables(self) -> list[Table]:

        # GROBID emits tables as <figure type="table"> with the grid in a nested <table>.
        tables = []
        for figure_element in self._find_all_by_local_name(self.root, 'figure'):
            if not self._is_table_figure(figure_element):
                continue

            table_element = self._find_first_by_local_name(figure_element, 'table')
            row_elements = (
                self._find_all_by_local_name(table_element, 'row')
                if table_element is not None
                else []
            )

            rows = [self._build_table_row(row_element) for row_element in row_elements]
            header_row, content_rows = self._split_header_and_content_rows(rows)

            tables.append(
                Table(
                    content_rows=content_rows,
                    header_row=header_row,
                    title=self._get_table_title(figure_element),
                    coords=figure_element.get('coords', ''),
                )
            )

        return tables


    def _is_candidate_VLE_table(self, table: Table) -> bool:
        """Cheap keyword pre-filter """

        keywords = ('vle', 'vapor', 'vapour', 'liquid', 'equilibrium', 'x1', 'x1', 'y1', 'y1')
        haystack = ' '.join(
            [table.title, *(cell.content for cell in table.header_row.cells)]
        ).lower()

        return any(keyword in haystack for keyword in keywords)


    def get_VLE_data(self):

        tables = self._detect_tables()
        candidate_tables = [table for table in tables if self._is_candidate_VLE_table(table)]

        return {
            "tables": tables,
            "candidate_tables": candidate_tables,
        }