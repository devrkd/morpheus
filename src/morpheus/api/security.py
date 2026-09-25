"""Inbound authentication as a FastAPI dependency.

Every route that can spend a provider credential, or that can read a
transcript, depends on :data:`PrincipalDep`. There is no route that reaches
the session store or a provider without one.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..auth.principals import ANONYMOUS, Principal, PrincipalStore
from ..errors import AuthenticationRequiredError

_bearer = HTTPBearer(auto_error=False, description="Inbound harness API key (mh_…)")


def get_principal(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)] = None,
) -> Principal:
    store: PrincipalStore | None = request.app.state.principals

    if store is None:
        # Anonymous mode. Reachable only because startup verified the operator
        # asked for it explicitly; see api/app.py.
        return ANONYMOUS

    if credentials is None or not credentials.credentials:
        raise AuthenticationRequiredError(
            "This endpoint requires an inbound API key. Send it as 'Authorization: Bearer <key>'."
        )

    return store.authenticate(credentials.credentials)


PrincipalDep = Annotated[Principal, Depends(get_principal)]
