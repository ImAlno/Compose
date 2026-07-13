"""Sortable unique identifiers (ULID-style) for composeai.

A ``new_ulid()`` result is a 26-character Crockford Base32 string encoding
128 bits: a 48-bit Unix-millisecond timestamp followed by 80 bits of
cryptographically random data from :func:`os.urandom`. Because the
timestamp occupies the most significant bits, two IDs generated at least
1 millisecond apart sort lexicographically in time order -- useful for
run/span/message identifiers that should also read as roughly
chronological when listed.

Stdlib only; no third-party dependency.
"""

from __future__ import annotations

import os
import time

# Crockford's Base32 alphabet: excludes I, L, O, U to avoid visual ambiguity.
_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TIMESTAMP_BYTES = 6  # 48 bits
_RANDOM_BYTES = 10  # 80 bits
_ENCODED_LENGTH = 26  # 128 bits / 5 bits-per-char (rounded up)


def new_ulid() -> str:
    """Return a new 26-character, lexicographically time-sortable ID."""
    timestamp_ms = int(time.time() * 1000)
    ts_bytes = timestamp_ms.to_bytes(_TIMESTAMP_BYTES, byteorder="big")
    rand_bytes = os.urandom(_RANDOM_BYTES)
    return _encode(ts_bytes + rand_bytes)


def _encode(data: bytes) -> str:
    """Encode 16 raw bytes (128 bits) as 26 Crockford Base32 characters."""
    value = int.from_bytes(data, byteorder="big")
    chars = [""] * _ENCODED_LENGTH
    for i in range(_ENCODED_LENGTH - 1, -1, -1):
        chars[i] = _ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)
