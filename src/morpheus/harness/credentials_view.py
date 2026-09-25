"""A tiny protocol so error translation does not import the whole factory."""

from __future__ import annotations

from typing import Protocol


class CredentialView(Protocol):
    @property
    def detected(self) -> bool:
        """Whether a credential was found for this provider.

        Gates one narrow behaviour: a bare ``TypeError`` is reinterpreted as a
        configuration problem only when nothing was detected, so a real
        ``TypeError`` in our own code still surfaces as the bug it is.
        """
        ...
