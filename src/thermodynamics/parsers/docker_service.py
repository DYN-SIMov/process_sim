from __future__ import annotations

import shutil
import subprocess
import time

import requests


class DockerNotAvailableError(RuntimeError):
    """Raised when Docker is not available on the system."""


class ContainerisedService:
    """Starts / reuses a Docker container that exposes an HTTP service.

    This has nothing to do with PDFs: any backend that ships as a container
    (GROBID, Nougat, marker, ...) can reuse it instead of re-implementing the
    Docker lifecycle.
    """

    def __init__(
        self,
        image: str,
        *,
        container_name: str | None = None,
        host_port: int = 8080,
        container_port: int = 8070,
        detach: bool = True,
        remove: bool = True,
        verbose: bool = True,
    ) -> None:
        self.image = image
        self.container_name = container_name
        self.host_port = host_port
        self.container_port = container_port
        self.detach = detach
        self.remove = remove
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def _run_docker(
        self,
        arguments: list[str],
        *,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["docker", *arguments],
            check=check,
            capture_output=True,
            text=True,
        )

    def verify_docker_is_available(self) -> None:
        if shutil.which("docker") is None:
            raise DockerNotAvailableError(
                "Docker CLI is not installed or not available"
            )

    def container_exists(self) -> bool:
        if self.container_name is None:
            return False

        inspect_process = self._run_docker(
            ["container", "inspect", self.container_name]
        )
        exists = inspect_process.returncode == 0

        if exists:
            self._log(f"Container '{self.container_name}' exists.")
        else:
            self._log(
                f"Could not find container '{self.container_name}'. "
                f"Error: {inspect_process.stderr}"
            )

        return exists

    def is_running(self) -> bool:
        if self.container_name is None:
            return False

        inspect_process = self._run_docker(
            [
                "container",
                "inspect",
                "--format",
                "{{.State.Running}}",
                self.container_name,
            ]
        )

        return (
            inspect_process.returncode == 0
            and inspect_process.stdout.strip().lower() == "true"
        )

    def published_host_port(self) -> int | None:
        if self.container_name is None:
            return None

        port_template = (
            f'{{{{(index (index .NetworkSettings.Ports '
            f'"{self.container_port}/tcp") 0).HostPort}}}}'
        )
        inspect_process = self._run_docker(
            [
                "container",
                "inspect",
                "--format",
                port_template,
                self.container_name,
            ]
        )

        published_port = inspect_process.stdout.strip()
        if not published_port:
            return None

        try:
            return int(published_port)
        except ValueError:
            return None

    def _start_existing_container(self) -> None:
        self._log(f"Starting existing container '{self.container_name}'...")

        arguments = ["start"]
        if not self.detach:
            arguments.append("-a")
        arguments.append(str(self.container_name))

        self._run_docker(arguments, check=True)

    def _run_new_container(self) -> None:
        self._log(f"Running new container from image '{self.image}'...")

        arguments = ["run"]
        if self.remove:
            arguments.append("--rm")
        if self.detach:
            arguments.append("-d")
        if self.container_name:
            arguments.extend(["--name", self.container_name])
        arguments.extend(["-p", f"{self.host_port}:{self.container_port}"])
        arguments.append(self.image)

        self._run_docker(arguments, check=True)

    def ensure_running(self) -> int:
        """Make sure the service is up and return the reachable host port."""

        self.verify_docker_is_available()

        if not self.container_exists():
            self._run_new_container()
            return self.host_port

        resolved_port = self.published_host_port() or self.host_port
        if self.is_running():
            self._log(f"Container '{self.container_name}' is already running.")
            return resolved_port

        self._start_existing_container()
        return resolved_port

    def stop(self) -> None:
        if self.container_name and self.is_running():
            self._run_docker(["stop", self.container_name])


def wait_until_http_ready(
    url: str,
    *,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 1.0,
    verbose: bool = True,
) -> None:
    started_at = time.monotonic()
    deadline = started_at + timeout_seconds

    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=2.0)
            if response.ok:
                if verbose:
                    elapsed = time.monotonic() - started_at
                    print(f"Service ready at {url} (took {elapsed:.2f} seconds)")
                return
        except requests.RequestException:
            pass

        time.sleep(poll_interval_seconds)

    raise TimeoutError(
        f"Service did not become ready at {url} within {timeout_seconds} seconds"
    )
