"""Download SAFE products using the Copernicus Data Space OData API."""

import logging
import os
import tempfile
import threading
import time
import zipfile
from urllib.parse import urljoin, urlparse

import requests

CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
logger = logging.getLogger("downloader")


class CopernicusDataSpace:
    """Share a short-lived access token across concurrent product downloads."""

    def __init__(self):
        self.username = os.environ.get("COPERNICUS_USERNAME")
        self.password = os.environ.get("COPERNICUS_PASSWORD")
        if not self.username or not self.password:
            raise ValueError(
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD to your "
                "Copernicus Data Space (https://dataspace.copernicus.eu/) credentials."
            )
        self._token = None
        self._expires_at = 0
        self._token_lock = threading.Lock()

    def _access_token(self, rejected_token=None):
        """Renew expired/rejected tokens without parallel authentication requests."""
        with self._token_lock:
            if time.monotonic() >= self._expires_at or self._token == rejected_token:
                with requests.post(
                    TOKEN_URL,
                    data={
                        "client_id": "cdse-public",
                        "grant_type": "password",
                        "username": self.username,
                        "password": self.password,
                    },
                    timeout=60,
                ) as response:
                    if response.status_code in (400, 401):
                        raise RuntimeError(
                            "Copernicus Data Space authentication failed. Check your "
                            "COPERNICUS_USERNAME and COPERNICUS_PASSWORD."
                        )
                    response.raise_for_status()
                    token = response.json()
                self._token = token["access_token"]
                self._expires_at = time.monotonic() + max(0, token["expires_in"] - 30)
            return self._token

    def find_product(self, name):
        """Resolve a SAFE name; legacy SciHub/database UUIDs are not CDSE IDs."""
        escaped_name = name.replace("'", "''")
        with requests.get(
            CATALOGUE_URL,
            params={"$filter": f"Name eq '{escaped_name}'", "$top": 2},
            timeout=60,
        ) as response:
            response.raise_for_status()
            products = response.json()["value"]
        if not products:
            raise RuntimeError(f"Product {name} not found in Copernicus Data Space.")
        if len(products) != 1:
            raise RuntimeError(f"Multiple Copernicus Data Space products match {name}.")
        product = products[0]
        if not product.get("Online", True):
            raise RuntimeError(f"Product {name} is offline in Copernicus Data Space.")
        return product["Id"]

    def _download_response(self, product_id):
        """Follow CDSE redirects, retaining authorization only on CDSE hosts."""
        url = f"{DOWNLOAD_URL}({product_id})/$value"
        token = self._access_token()
        renewed = False
        for _ in range(10):
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {token}"},
                stream=True,
                allow_redirects=False,
                timeout=(60, 120),
            )
            if response.status_code == 401 and not renewed:
                response.close()
                token = self._access_token(rejected_token=token)
                renewed = True
                continue
            if response.is_redirect:
                location = urljoin(url, response.headers["Location"])
                response.close()
                parsed = urlparse(location)
                if parsed.scheme != "https" or not (parsed.hostname or "").endswith(
                    ".dataspace.copernicus.eu"
                ):
                    raise RuntimeError("Unexpected Copernicus Data Space download redirect.")
                url = location
                continue
            try:
                response.raise_for_status()
            except requests.RequestException:
                response.close()
                raise
            return response
        raise RuntimeError("Too many Copernicus Data Space download redirects.")

    def download(self, name, directory):
        """Stream an archive to a temporary file and publish only a complete ZIP."""
        os.makedirs(directory, exist_ok=True)
        destination = os.path.join(directory, name.replace(".SAFE", ".zip"))
        product_id = self.find_product(name)
        for attempt in range(4):
            partial = None
            try:
                with tempfile.NamedTemporaryFile(
                    dir=directory, suffix=".part", delete=False
                ) as output:
                    partial = output.name
                    with self._download_response(product_id) as response:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            output.write(chunk)
                if not zipfile.is_zipfile(partial):
                    raise RuntimeError(f"Downloaded product {name} is not a ZIP archive.")
                os.replace(partial, destination)
                logger.info("Downloaded %s", name)
                return
            except requests.RequestException as error:
                status = error.response.status_code if error.response is not None else None
                if attempt == 3 or (status and status < 500 and status != 429):
                    raise
                time.sleep(2 ** attempt)
            finally:
                if partial and os.path.exists(partial):
                    os.remove(partial)
