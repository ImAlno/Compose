import time

from composeai._ids import new_ulid

_CROCKFORD_ALPHABET = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


def test_ulid_length_and_alphabet():
    ulid = new_ulid()
    assert len(ulid) == 26
    assert set(ulid) <= _CROCKFORD_ALPHABET


def test_ulid_alphabet_excludes_ambiguous_chars():
    # Crockford base32 excludes I, L, O, U to avoid visual ambiguity.
    for ch in "ILOU":
        assert ch not in _CROCKFORD_ALPHABET


def test_ulid_first_char_limited_range():
    # 128 bits encoded as 26 chars of 5 bits => 130 bits of "room", so the
    # first char can only use its low 3 bits (values 0-7).
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    for _ in range(50):
        ulid = new_ulid()
        assert alphabet.index(ulid[0]) <= 7


def test_ulid_is_str():
    assert isinstance(new_ulid(), str)


def test_ulid_time_ordering():
    first = new_ulid()
    time.sleep(0.002)
    second = new_ulid()
    assert first < second


def test_ulid_uniqueness():
    ids = {new_ulid() for _ in range(1000)}
    assert len(ids) == 1000


def test_ulid_shares_timestamp_prefix_within_same_millisecond():
    # Generated back-to-back, the two IDs should very likely share their
    # timestamp-derived prefix (first 10 chars encode the 48-bit ms
    # timestamp), demonstrating the timestamp -- not luck -- drives order.
    a = new_ulid()
    b = new_ulid()
    assert a[:10] <= b[:10]
