import pytest

from composeai.errors import (
    BudgetExceededError,
    ComposeError,
    CompositionTypeError,
    ConfigError,
    ProviderError,
    ResumeMismatchError,
    SerializationError,
)

ALL_ERRORS = [
    ComposeError,
    ConfigError,
    ProviderError,
    SerializationError,
    ResumeMismatchError,
    BudgetExceededError,
    CompositionTypeError,
]


def test_compose_error_is_exception():
    assert issubclass(ComposeError, Exception)


@pytest.mark.parametrize(
    "exc_cls",
    [
        ConfigError,
        ProviderError,
        SerializationError,
        ResumeMismatchError,
        BudgetExceededError,
        CompositionTypeError,
    ],
)
def test_subclasses_of_compose_error(exc_cls):
    assert issubclass(exc_cls, ComposeError)


def test_composition_type_error_is_also_type_error():
    assert issubclass(CompositionTypeError, TypeError)
    assert issubclass(CompositionTypeError, ComposeError)
    with pytest.raises(TypeError):
        raise CompositionTypeError("pipe() output type does not match input type")


def test_composition_type_error_catchable_as_compose_error():
    with pytest.raises(ComposeError):
        raise CompositionTypeError("mismatch")


def test_provider_error_attrs_default_to_none():
    err = ProviderError("boom")
    assert err.provider is None
    assert err.model is None
    assert str(err) == "boom"


def test_provider_error_attrs_are_keyword_only():
    err = ProviderError("boom", provider="anthropic", model="claude-sonnet-5")
    assert err.provider == "anthropic"
    assert err.model == "claude-sonnet-5"


def test_provider_error_rejects_positional_attrs():
    with pytest.raises(TypeError):
        ProviderError("boom", "anthropic", "claude-sonnet-5")  # type: ignore[misc]


def test_all_error_classes_have_docstrings():
    for exc_cls in ALL_ERRORS:
        assert exc_cls.__doc__ and exc_cls.__doc__.strip(), f"{exc_cls} missing docstring"


def test_errors_carry_no_extra_frames_beyond_raise_site():
    # Design goal: raising/catching one of our errors shouldn't add deep
    # framework frames on top of the user's own raise site.
    try:
        raise ConfigError("missing ANTHROPIC_API_KEY")
    except ConfigError as exc:
        tb = exc.__traceback__
        frame_count = 0
        while tb is not None:
            frame_count += 1
            tb = tb.tb_next
        assert frame_count == 1
