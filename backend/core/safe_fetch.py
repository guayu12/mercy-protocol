"""Bounded HTTP media fetching with DNS validation at every TCP connection."""

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from core.file_cleanup import discard_file


CONNECT_TIMEOUT = 3.0
READ_TIMEOUT = 10.0
MAX_REDIRECTS = 3
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
NAT64_PREFIXES = (
    ipaddress.IPv6Network("64:ff9b::/96"),
    ipaddress.IPv6Network("64:ff9b:1::/48"),
)


class SafeFetchError(Exception):
    pass


class UnsafeAddress(SafeFetchError):
    pass


class DownloadTooLarge(SafeFetchError):
    pass


_HTTPCORE_TO_HTTPX = (
    (httpcore.ConnectTimeout, httpx.ConnectTimeout),
    (httpcore.ReadTimeout, httpx.ReadTimeout),
    (httpcore.WriteTimeout, httpx.WriteTimeout),
    (httpcore.PoolTimeout, httpx.PoolTimeout),
    (httpcore.ConnectError, httpx.ConnectError),
    (httpcore.ReadError, httpx.ReadError),
    (httpcore.WriteError, httpx.WriteError),
    (httpcore.RemoteProtocolError, httpx.RemoteProtocolError),
    (httpcore.LocalProtocolError, httpx.LocalProtocolError),
    (httpcore.UnsupportedProtocol, httpx.UnsupportedProtocol),
    (httpcore.ProxyError, httpx.ProxyError),
    (httpcore.ProtocolError, httpx.ProtocolError),
    (httpcore.NetworkError, httpx.NetworkError),
    (httpcore.TimeoutException, httpx.TimeoutException),
)
_HTTPCORE_ERRORS = tuple(error_type for error_type, _ in _HTTPCORE_TO_HTTPX)


def _as_httpx_error(error: Exception, request: httpx.Request) -> httpx.TransportError:
    for core_type, httpx_type in _HTTPCORE_TO_HTTPX:
        if isinstance(error, core_type):
            return httpx_type(str(error), request=request)
    return httpx.TransportError(str(error), request=request)


def _is_public_address(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address):
        if address in NAT64_PREFIXES[0]:
            address = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
        elif address in NAT64_PREFIXES[1]:
            value = int(address)
            embedded_ipv4 = ((value >> 64) & 0xFFFF) << 16
            embedded_ipv4 |= (value >> 40) & 0xFFFF
            address = ipaddress.IPv4Address(embedded_ipv4)
        elif address.ipv4_mapped is not None:
            address = address.ipv4_mapped
        elif int(address) >> 32 == 0 and int(address) not in (0, 1):
            embedded = ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
            if not embedded.is_global or embedded.is_multicast:
                return False
        if address.version == 6:
            if address.sixtofour is not None and (
                not address.sixtofour.is_global or address.sixtofour.is_multicast
            ):
                return False
            if address.teredo is not None:
                server, client = address.teredo
                if any(not item.is_global or item.is_multicast for item in (server, client)):
                    return False
    return address.is_global and not address.is_multicast


async def _resolve_addresses(host: str, port: int) -> list[str]:
    loop = asyncio.get_running_loop()
    results = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(result[4][0] for result in results))


async def _validated_ip(
    host: str,
    port: int,
    resolver=_resolve_addresses,
) -> str:
    try:
        addresses = await resolver(host, port)
    except Exception as error:
        raise UnsafeAddress(f"Could not resolve media host: {host}") from error
    if not addresses:
        raise UnsafeAddress(f"Media host has no addresses: {host}")
    parsed = []
    try:
        parsed = [ipaddress.ip_address(item.split("%", 1)[0]) for item in addresses]
    except ValueError as error:
        raise UnsafeAddress(f"Media host returned an invalid address: {host}") from error
    if any(not _is_public_address(str(address)) for address in parsed):
        raise UnsafeAddress(f"Media host resolves to a non-public address: {host}")
    return str(parsed[0])


def _parse_target(url: str) -> tuple[str, int]:
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError as error:
        raise SafeFetchError("Invalid media URL") from error
    if parsed.scheme.lower() not in {"http", "https"}:
        raise SafeFetchError("Only HTTP and HTTPS media URLs are allowed")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise SafeFetchError("Invalid media URL")
    return parsed.hostname, port or (443 if parsed.scheme.lower() == "https" else 80)


async def _validate_target(url: str, resolver=_resolve_addresses) -> None:
    host, port = _parse_target(url)
    await _validated_ip(host, port, resolver)


class ValidatingNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, resolver=_resolve_addresses, backend=None):
        self._resolver = resolver
        self._backend = backend or httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options=None,
    ):
        address = await _validated_ip(host, port, self._resolver)
        return await self._backend.connect_tcp(
            host=address,
            port=port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise httpcore.ConnectError("Unix sockets are disabled for media fetching")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _CoreResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream, request: httpx.Request):
        self._stream = stream
        self._request = request

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                yield chunk
        except _HTTPCORE_ERRORS as error:
            raise _as_httpx_error(error, self._request) from error

    async def aclose(self) -> None:
        await self._stream.aclose()


class SafeHTTPXTransport(httpx.AsyncBaseTransport):
    def __init__(self, resolver=_resolve_addresses):
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=ValidatingNetworkBackend(resolver=resolver)
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        try:
            core_response = await self._pool.handle_async_request(core_request)
        except _HTTPCORE_ERRORS as error:
            raise _as_httpx_error(error, request) from error
        return httpx.Response(
            status_code=core_response.status,
            headers=core_response.headers,
            stream=_CoreResponseStream(core_response.stream, request),
            extensions=core_response.extensions,
            request=request,
        )

    async def aclose(self) -> None:
        await self._pool.aclose()


@dataclass
class FetchStream:
    client: httpx.AsyncClient
    response: httpx.Response
    max_bytes: int

    async def aiter_bytes(self):
        total = 0
        async for chunk in self.response.aiter_raw():
            total += len(chunk)
            if total > self.max_bytes:
                raise DownloadTooLarge("Remote media exceeds size limit")
            yield chunk

    async def aclose(self) -> None:
        await self.response.aclose()
        await self.client.aclose()


class SafeMediaFetcher:
    def __init__(self, resolver=_resolve_addresses, transport_factory=None):
        self._resolver = resolver
        self._transport_factory = transport_factory

    def _client(self) -> httpx.AsyncClient:
        transport = (
            self._transport_factory()
            if self._transport_factory
            else SafeHTTPXTransport(resolver=self._resolver)
        )
        return httpx.AsyncClient(
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT,
                read=READ_TIMEOUT,
                write=READ_TIMEOUT,
                pool=CONNECT_TIMEOUT,
            ),
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": (
                    "MercyProtocol/1.0 "
                    "(+https://github.com/guayu12/mercy-protocol)"
                ),
            },
        )

    async def _request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        current_url = url
        for redirect_count in range(MAX_REDIRECTS + 1):
            await _validate_target(current_url, self._resolver)
            request = client.build_request(method, current_url, headers=headers)
            try:
                response = await client.send(
                    request, stream=True, follow_redirects=False
                )
            except httpx.HTTPError as error:
                raise SafeFetchError("Remote media request failed") from error
            if response.status_code not in REDIRECT_STATUSES:
                return response
            location = response.headers.get("location")
            await response.aclose()
            if not location:
                raise SafeFetchError("Media redirect has no Location header")
            if redirect_count >= MAX_REDIRECTS:
                raise SafeFetchError("Too many media redirects")
            current_url = urljoin(current_url, location)
        raise SafeFetchError("Too many media redirects")

    async def probe(self, url: str) -> dict:
        async with self._client() as client:
            try:
                response = await self._request(client, "HEAD", url)
            except UnsafeAddress:
                raise
            except SafeFetchError:
                response = None
            if response is None or not 200 <= response.status_code < 300:
                if response is not None:
                    await response.aclose()
                response = await self._request(
                    client, "GET", url, headers={"Range": "bytes=0-0"}
                )
            try:
                headers = response.headers
                length = _response_length(headers)
                return {
                    "status": response.status_code,
                    "content_type": headers.get("content-type", "").split(";", 1)[0].strip().lower(),
                    "length": length,
                }
            finally:
                await response.aclose()

    async def download_to_file(self, url: str, path: str | Path, max_bytes: int) -> dict:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        async with self._client() as client:
            response = await self._request(client, "GET", url)
            total = 0
            try:
                if response.status_code != 200:
                    raise SafeFetchError(f"Remote media returned HTTP {response.status_code}")
                length = _response_length(response.headers)
                if length is not None and length > max_bytes:
                    raise DownloadTooLarge("Remote media exceeds size limit")
                with destination.open("wb") as output:
                    async for chunk in response.aiter_raw():
                        total += len(chunk)
                        if total > max_bytes:
                            raise DownloadTooLarge("Remote media exceeds size limit")
                        output.write(chunk)
                return {
                    "status": response.status_code,
                    "content_type": response.headers.get("content-type", "").split(";", 1)[0].strip().lower(),
                    "length": total,
                }
            except Exception:
                discard_file(destination)
                raise
            finally:
                await response.aclose()

    async def open_stream(
        self, url: str, range_header: str | None, max_bytes: int
    ) -> FetchStream:
        client = self._client()
        try:
            headers = {"Range": range_header} if range_header else None
            response = await self._request(client, "GET", url, headers=headers)
            return FetchStream(client, response, max_bytes)
        except Exception:
            await client.aclose()
            raise


def _response_length(headers: httpx.Headers) -> int | None:
    content_range = headers.get("content-range", "")
    if "/" in content_range:
        total = content_range.rsplit("/", 1)[-1].strip()
        if total.isdigit():
            return int(total)
    value = headers.get("content-length")
    return int(value) if value and value.isdigit() else None


safe_media_fetcher = SafeMediaFetcher()
