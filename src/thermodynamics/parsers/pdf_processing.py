from __future__ import annotations
import shutil
import subprocess
from pathlib import Path

import requests

from thermodynamics.parsers.xml_parser import XMLParser

class DockerNotAvailableError(RuntimeError):
    """Raised when Docker is not available on the system."""
    pass

class GrobidError(RuntimeError):
    """Raised when there is an error interacting with the GROBID service."""
    pass


class PDFParser(): 

    def __init__(self, 
                 xml_parsing_engine: XMLParser | None = None,
                 verbose: bool = True): 

        self.verbose = verbose
        self.docker_timeout_seconds = 300.0


    def _verify_docker_is_available(self) -> None: 
        if shutil.which("docker") is None:
            raise DockerNotAvailableError("Docker CLI is not installed or not available")

    
    def _check_if_container_exists(self, container_name: str | None) -> bool:

        if container_name is None:
            return False

        inspect_process = subprocess.run(
            ["docker", "container", "inspect", container_name],
            check=False,
            capture_output=True,
            text=True,
        )

        if self.verbose:
            if inspect_process.returncode == 0:
                print(f"Container '{container_name}' exists.")
            else:
                print(f"Could not find container '{container_name}'. Error: {inspect_process.stderr}")

        return inspect_process.returncode == 0


    def _get_container_host_port(self, container_name: str, container_port: int) -> int | None:

        inspect_process = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                f"{{{{(index (index .NetworkSettings.Ports \"{container_port}/tcp\") 0).HostPort}}}}",
                container_name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        host_port = inspect_process.stdout.strip()
        if not host_port:
            return None

        try:
            return int(host_port)
        except ValueError:
            return None


    def _is_container_running(self, container_name: str | None) -> bool:

        if container_name is None:
            return False

        inspect_process = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                container_name,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        is_running = inspect_process.returncode == 0 and inspect_process.stdout.strip().lower() == "true"
        if self.verbose:
            if inspect_process.returncode != 0:
                print(f"Could not inspect running state for container '{container_name}'. Error: {inspect_process.stderr}")
            elif is_running:
                print(f"Container '{container_name}' is already running.")
            else:
                print(f"Container '{container_name}' exists but is not running.")

        return is_running


    def _get_running_container(self,
                               container_name: str | None = None,
                               resolved_port: int | None = None,
                               dry_run: bool = False) -> None:

        if dry_run:
            docker_command = [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                container_name,
            ]
            print(f"Dry run: {' '.join(docker_command)}")
            return docker_command, resolved_port

        inspect_process = subprocess.run(
            [
                "docker",
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                container_name,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return inspect_process, resolved_port


    def _start_existing_container(self, 
                                  container_name: str, 
                                  detach: bool = True,
                                  dry_run: bool = False) -> list[str] | subprocess.CompletedProcess[str]:

        docker_command = ["docker", "start"]
        if not detach:
            docker_command.append("-a")
        docker_command.append(container_name)

        if self.verbose and not dry_run:
            print(f"Starting existing container '{container_name}'...")

        if dry_run:
            print(f"Dry run: {' '.join(docker_command)}")
            return docker_command

        return subprocess.run(
            docker_command,
            check=True,
            capture_output=True,
            text=True,
        )


    def _run_new_container(self,
                           image: str,
                           container_name: str | None = None,
                           local_port: int | None = None,
                           container_port: int | None = None,
                           detach: bool = True,
                           remove: bool = True,
                           dry_run: bool = False) -> list[str] | subprocess.CompletedProcess[str]:

        docker_command: list[str] = ["docker", "run"]

        if remove:
            docker_command.append("--rm")
        if detach:
            docker_command.append("-d")
        if container_name:
            docker_command.extend(["--name", container_name])
        if local_port is not None:
            docker_command.extend(["-p", f"{local_port}:{container_port or local_port}"])

        docker_command.append(image)

        if self.verbose and not dry_run:
            print(f"Running new container from image '{image}'...")

        if dry_run:
            print(f"Dry run: {' '.join(docker_command)}")
            return docker_command

        return subprocess.run(
            docker_command,
            check=True,
            capture_output=True,
            text=True,
        )


    def _initiate_docker_container(
        self,
        image: str,
        *,
        container_name: str | None = None,
        local_port: int = 8080,
        container_port: int = 8070,
        detach: bool = True,
        remove: bool = True,
        dry_run: bool = True,
    ) -> tuple[list[str] | subprocess.CompletedProcess[str], int]:

        """
        Initiates or reuses a Docker container.

        Parameters:
            image (str): The Docker image to run.
            container_name (str | None): The name of the container. If a container with this
                name already exists, this method starts it instead of running a new image.
            local_port (int): The host port to bind when creating a new container.
            container_port (int): The port exposed by the container.
            detach (bool): Whether to run the container in detached mode.
            remove (bool): Whether to remove the container after it exits when creating a new
                container with docker run.
            dry_run (bool): If True, return the Docker command instead of executing it.

        Returns:
            tuple[list[str] | subprocess.CompletedProcess[str], int]: The Docker command if
            dry_run is True, otherwise the result of subprocess.run, together with the host port
            that should be used to reach the container.
        """

        if self._check_if_container_exists(container_name): 
            resolved_port = self._get_container_host_port(container_name, container_port) or local_port

            if self._is_container_running(container_name):
                subprocess_result, resolved_port = self._get_running_container(
                    container_name=container_name,
                    resolved_port=resolved_port,
                    dry_run=dry_run
                )
                return subprocess_result, resolved_port

            subprocess_result = self._start_existing_container(
                container_name=container_name, 
                detach=detach, 
                dry_run=dry_run
            )
            return subprocess_result, resolved_port

        subprocess_result = self._run_new_container(
            image=image,
            container_name=container_name,
            local_port=local_port,
            container_port=container_port,
            detach=detach,
            remove=remove,
            dry_run=dry_run,
        )
        return subprocess_result, local_port


class VLEDataPDFParser(PDFParser): 

    def __init__(self): 
        super().__init__()

        self._verify_docker_is_available()

        self.grobid, self.grobid_port = self._initiate_docker_container(
            image="grobid/grobid:0.9.0-full",
            container_name="pavels_grobid",
            local_port=8070,
            container_port=8070,
            detach=True,
            remove=True,
            dry_run=False,
        )
        self.grobid_base_url = f"http://localhost:{self.grobid_port}"

        self._wait_for_grobid_to_be_ready(timeout_seconds=self.docker_timeout_seconds)


    def _wait_for_grobid_to_be_ready(self, timeout_seconds: float = 120.0) -> None:

        ready_url = self.grobid_base_url + "/api/isalive"
        deadline = __import__("time").monotonic() + timeout_seconds

        time_started = __import__("time").monotonic()
        while __import__("time").monotonic() < deadline:
            try:
                response = requests.get(ready_url, timeout=2.0)
                if response.ok:
                    if self.verbose:
                        time_it_took = __import__("time").monotonic() - time_started
                        print(f"GROBID is ready at {ready_url} (took {time_it_took:.2f} seconds)")
                    return
            except requests.RequestException:
                pass

            __import__("time").sleep(1.0)

        raise TimeoutError(
            f"GROBID did not become ready at {ready_url} within {timeout_seconds} seconds"
        )


    def _send_pdf_to_grobid_via_http(self, pdf_path: str | Path) -> dict:

        """  
        Ref general: https://grobid.readthedocs.io/en/latest/Grobid-service/
        Ref tei coordinates: https://grobid.readthedocs.io/en/latest/Coordinates-in-PDF/
        """

        consolidate_header = "1" 
        tei_coordinates_items = ["s", "ref", "biblStruct", "figure"]

        grobid_config: list[tuple[str, str]] = [
            ("consolidateHeader", consolidate_header),
        ]

        for item in (tei_coordinates_items or []):
            grobid_config.append(("teiCoordinates", item))

        with open(pdf_path, "rb") as f:
            response = requests.post(
                self.grobid_base_url + "/api/processFulltextDocument",
                files={"input": (Path(pdf_path).name, f, "application/pdf")},
                data=grobid_config,
                timeout=self.docker_timeout_seconds,
            )

        status = response.status_code
        if status != 200:
            raise GrobidError(
                f"GROBID returned an unexpected status code {status} for PDF '{pdf_path}'"
            )

        response.raise_for_status()
        tei_xml = response.text
        
        return {"tei_xml": tei_xml, "status_code": status}



    def _extract_VLE_data_from_tei_xml(self, tei_xml: str) -> dict:

        xml_parser = XMLParser(xml_file=tei_xml)
        metadata = xml_parser.get_metadata()
        vle_data = xml_parser.get_VLE_data()

        return {"metadata": metadata, "vle_data": vle_data}



    def get_VLE_data_from_pdf(self, pdf_path: str | Path) -> None: 

        grobid_output = self._send_pdf_to_grobid_via_http(pdf_path)
        tei_xml = grobid_output["tei_xml"]
        vle_data_extract = self._extract_VLE_data_from_tei_xml(tei_xml)

        print(vle_data_extract)

        return vle_data_extract











