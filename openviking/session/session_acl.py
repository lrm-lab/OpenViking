# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Session-owner visibility labels for causal retrieval objects.

The label is ``user:{user_id}`` only. Group labels are not issued here.
Owner resolution order is fixed: ``ctx.user.user_id``, then
``_meta.created_by_user_id``, then the user segment of ``_session_uri``
(``viking://user/<user_id>/sessions/<id>``). No owner means the object is
not visible.
"""

from __future__ import annotations

from typing import Any

from openviking.session.influence_projection import SessionAcl

_SCHEME = "viking://"


def session_owner_label(session: Any) -> str:
    """Return the single owner label for ``session``.

    Raises ``ValueError`` when none of the three owner sources yields a
    non-empty user id.
    """
    user_id = _owner_user_id(session)
    if user_id is None:
        raise ValueError("session owner could not be resolved")
    return f"user:{user_id}"


def session_stored_owner_label(session: Any) -> str:
    """Owner from ``_meta`` or ``_session_uri``. Caller ctx is ignored.

    ``session_owner_label`` prefers ``ctx.user.user_id``. The diagnose HTTP
    face loads a Session with the caller's ctx, so that helper would always
    return the caller. Target visibility must use the stored owner.
    """
    user_id = _user_id_from_meta(session) or _user_id_from_session_uri(
        getattr(session, "_session_uri", None)
    )
    if user_id is None:
        raise ValueError("session owner could not be resolved")
    return f"user:{user_id}"


def session_acl_callback(session: Any) -> SessionAcl:
    """Return a callback that labels only this session.

    Any other session id raises ``ValueError``. The callback does not return
    an empty sequence, so a registry bound to the wrong session fails closed.
    """
    label = session_owner_label(session)
    session_id = _session_id(session)

    def _labels(queried_session_id: str) -> tuple[str, ...]:
        if queried_session_id != session_id:
            raise ValueError(
                f"session ACL callback is bound to {session_id!r}, refusing {queried_session_id!r}"
            )
        return (label,)

    return _labels


def _owner_user_id(session: Any) -> str | None:
    return (
        _user_id_from_ctx(session)
        or _user_id_from_meta(session)
        or _user_id_from_session_uri(getattr(session, "_session_uri", None))
    )


def _session_id(session: Any) -> str:
    session_id = _nonempty_str(getattr(session, "session_id", None))
    if session_id is None:
        raise ValueError("session_id is required to bind a session ACL callback")
    return session_id


def _user_id_from_ctx(session: Any) -> str | None:
    ctx = getattr(session, "ctx", None)
    if ctx is None:
        return None
    user = getattr(ctx, "user", None)
    if user is None:
        return None
    return _nonempty_str(getattr(user, "user_id", None))


def _user_id_from_meta(session: Any) -> str | None:
    meta = getattr(session, "_meta", None)
    if meta is None:
        return None
    return _nonempty_str(getattr(meta, "created_by_user_id", None))


def _user_id_from_session_uri(uri: Any) -> str | None:
    if not isinstance(uri, str):
        return None
    text = uri.split("?", 1)[0].strip()
    if not text.startswith(_SCHEME):
        return None
    parts = [part for part in text[len(_SCHEME) :].split("/") if part]
    if len(parts) < 4 or parts[0] != "user" or parts[2] != "sessions":
        return None
    return _nonempty_str(parts[1])


def _nonempty_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    return text
