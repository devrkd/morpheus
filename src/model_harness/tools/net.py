"""Address validation for outbound tool requests.

A model-driven HTTP tool is a server-side request forgery engine unless the
destination is checked. The model decides the URL, and anything the *server*
can reach becomes reachable — cloud instance metadata at 169.254.169.254,
internal admin panels, a database on a private subnet, or this very harness on
loopback, where the tool could read other people's sessions.

So the default is deny: only public, non-loopback, non-private addresses. And
validation happens against the *resolved addresses*, not the hostname, because
a name under an attacker's control (or simply a misconfigured internal record)
can point anywhere — including ``localhost.example.com`` resolving to
127.0.0.1. Every redirect hop is revalidated for the same reason.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = frozenset({"http", "https"})


class AddressRejected(Exception):
    """The destination is not allowed. The message is shown to the model."""


def _classify(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> str | None:
    """Return a rejection reason, or None when the address is acceptable."""
    if ip.is_unspecified:
        # 0.0.0.0 / :: — also reported as private by ipaddress, so it is
        # checked first to give an accurate reason.
        return "the unspecified (reserved) address"
    if ip.is_loopback:
        return "a loopback address"
    if ip.is_link_local:
        # Covers 169.254.0.0/16, which is where cloud instance metadata and
        # its credentials live.
        return "a link-local address (cloud instance metadata lives here)"
    if ip.is_private:
        return "a private address"
    if ip.is_multicast:
        return "a multicast address"
    if ip.is_reserved:
        return "a reserved address"

    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        # ::ffff:127.0.0.1 must not slip past the IPv4 checks above.
        return _classify(mapped)

    return None


async def resolve_and_check(url: str, *, allowed_hosts: frozenset[str] | None = None) -> str:
    """Validate a URL for outbound fetching. Returns its hostname.

    Raises :class:`AddressRejected` with a model-readable explanation.
    """
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        raise AddressRejected(
            f"Scheme '{parsed.scheme or '(none)'}' is not allowed. Use http or https."
        )

    host = parsed.hostname
    if not host:
        raise AddressRejected(f"Could not read a hostname from '{url}'.")

    if allowed_hosts is not None and host.lower() not in allowed_hosts:
        raise AddressRejected(
            f"Host '{host}' is not on this deployment's allowlist of {sorted(allowed_hosts)}."
        )

    # A bare IP literal is checked directly; anything else is resolved first,
    # and every address it resolves to must pass.
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        reason = _classify(literal)
        if reason:
            raise AddressRejected(f"{host} is {reason}, which is not reachable from here.")
        return host

    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise AddressRejected(f"Could not resolve '{host}': {exc.strerror or exc}.") from exc

    if not infos:
        raise AddressRejected(f"'{host}' did not resolve to any address.")

    for info in infos:
        address = info[4][0]
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        reason = _classify(ip)
        if reason:
            raise AddressRejected(
                f"'{host}' resolves to {address}, which is {reason}. Requests to "
                "internal addresses are blocked."
            )

    return host
