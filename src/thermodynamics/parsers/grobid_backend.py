from __future__ import annotations

from pathlib import Path

import requests

from thermodynamics.parsers.base import BackendError, RawPayloadParser
from thermodynamics.parsers.docker_service import (
    ContainerisedService,
    wait_until_http_ready,
)
from thermodynamics.parsers.document_model import ParsedDocument
from thermodynamics.parsers.xml_parser import TEIXMLParser


class GrobidBackend:
    """PDF extraction through a GROBID service.

    The service can either be managed locally through Docker (default) or be an
    already running instance reachable at `base_url`.

    Refs:
        https://grobid.readthedocs.io/en/latest/Grobid-service/
        https://grobid.readthedocs.io/en/latest/Coordinates-in-PDF/
    """

    def __init__(
        self,
        *,
        name: str = "grobid",
        base_url: str | None = None,
        image: str = "grobid/grobid:0.9.0-full",
        container_name: str = "pavels_grobid",
        host_port: int = 8070,
        container_port: int = 8070,
        payload_parser: RawPayloadParser | None = None,
        request_timeout_seconds: float = 300.0,
        startup_timeout_seconds: float = 300.0,
        verbose: bool = True,
        allow_service_autostart: bool = False,
    ) -> None:

        self._name = name
        self.verbose = verbose
        self.request_timeout_seconds = request_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.allow_service_autostart = allow_service_autostart
        self.payload_parser = payload_parser or TEIXMLParser()

        self._base_url = base_url.rstrip("/") if base_url else None
        self._service = (
            None
            if base_url
            else ContainerisedService(
                image=image,
                container_name=container_name,
                host_port=host_port,
                container_port=container_port,
                verbose=verbose,
            )
        )

    @property
    def name(self) -> str:
        return self._name

    def _ensure_service_is_ready(self) -> str:
        """Start the container on first use, so constructing a backend is cheap."""

        if self._base_url is not None:
            return self._base_url

        if not self.allow_service_autostart:
            raise BackendError(
                "Refusing to auto-start local GROBID service without explicit "
                "permission. Set allow_service_autostart=True or provide base_url."
            )

        assert self._service is not None
        resolved_port = self._service.ensure_running()
        base_url = f"http://localhost:{resolved_port}"

        wait_until_http_ready(
            f"{base_url}/api/isalive",
            timeout_seconds=self.startup_timeout_seconds,
            verbose=self.verbose,
        )

        self._base_url = base_url
        return base_url

    def _request_tei_xml(self, pdf_path: str | Path) -> str:

        base_url = self._ensure_service_is_ready()

        grobid_config: list[tuple[str, str]] = [("consolidateHeader", "1")]
        for coordinates_item in ("s", "ref", "biblStruct", "figure"):
            grobid_config.append(("teiCoordinates", coordinates_item))

        with open(pdf_path, "rb") as pdf_file:
            response = requests.post(
                f"{base_url}/api/processFulltextDocument",
                files={"input": (Path(pdf_path).name, pdf_file, "application/pdf")},
                data=grobid_config,
                timeout=self.request_timeout_seconds,
            )

        if response.status_code != 200:
            raise BackendError(
                f"GROBID returned status {response.status_code} for PDF '{pdf_path}'"
            )

        return response.text

    def extract(self, pdf_path: str | Path) -> ParsedDocument:

        tei_xml = self._request_tei_xml(pdf_path)

        return self.payload_parser.parse(
            tei_xml,
            source_path=str(pdf_path),
            backend_name=self.name,
        )

    def close(self) -> None:
        """Containers started here are left running on purpose (reuse is cheap)."""
        return None
