import pytest
from pydantic import TypeAdapter, ValidationError

from composeai.messages import (
    ContentPart,
    ImagePart,
    Message,
    StopReason,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)

_PartAdapter: TypeAdapter[ContentPart] = TypeAdapter(ContentPart)


# --- ContentPart discriminated union ---


def test_text_part_from_dict():
    part = _PartAdapter.validate_python({"type": "text", "text": "hi"})
    assert isinstance(part, TextPart)
    assert part.text == "hi"


def test_tool_call_part_from_dict():
    part = _PartAdapter.validate_python(
        {"type": "tool_call", "id": "1", "name": "search", "arguments": {"q": "x"}}
    )
    assert isinstance(part, ToolCallPart)
    assert part.id == "1"
    assert part.name == "search"
    assert part.arguments == {"q": "x"}


def test_tool_result_part_from_dict_defaults_is_error_false():
    part = _PartAdapter.validate_python(
        {"type": "tool_result", "tool_call_id": "1", "content": "result"}
    )
    assert isinstance(part, ToolResultPart)
    assert part.is_error is False


def test_thinking_part_defaults():
    part = _PartAdapter.validate_python({"type": "thinking"})
    assert isinstance(part, ThinkingPart)
    assert part.text == ""
    assert part.provider_token is None
    assert part.provider is None


def test_thinking_part_with_provider_token():
    part = _PartAdapter.validate_python(
        {"type": "thinking", "text": "hmm", "provider_token": "tok", "provider": "anthropic"}
    )
    assert isinstance(part, ThinkingPart)
    assert part.provider_token == "tok"
    assert part.provider == "anthropic"


def test_content_part_round_trip_for_every_variant():
    payloads = [
        {"type": "text", "text": "hello"},
        {"type": "image", "media_type": "image/png", "data": "YWJj"},
        {"type": "image", "media_type": "image/png", "url": "https://example.com/x.png"},
        {"type": "tool_call", "id": "1", "name": "f", "arguments": {}},
        {"type": "tool_result", "tool_call_id": "1", "content": "ok", "is_error": True},
        {"type": "thinking", "text": "hmm"},
    ]
    for payload in payloads:
        part = _PartAdapter.validate_python(payload)
        dumped = _PartAdapter.dump_python(part, mode="json")
        part2 = _PartAdapter.validate_python(dumped)
        assert part == part2


def test_unknown_discriminator_raises():
    with pytest.raises(ValidationError):
        _PartAdapter.validate_python({"type": "bogus"})


# --- ImagePart exactly-one-source validation ---


def test_image_part_data_only_is_valid():
    part = ImagePart(media_type="image/png", data="YWJj")
    assert part.data == "YWJj"
    assert part.url is None


def test_image_part_url_only_is_valid():
    part = ImagePart(media_type="image/png", url="https://example.com/x.png")
    assert part.url == "https://example.com/x.png"
    assert part.data is None


def test_image_part_neither_source_raises():
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png")


def test_image_part_both_sources_raises():
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png", data="YWJj", url="https://example.com/x.png")


# --- Message conveniences ---


def test_message_user_from_str_makes_single_text_part():
    msg = Message.user("hello")
    assert msg.role == "user"
    assert len(msg.parts) == 1
    assert isinstance(msg.parts[0], TextPart)
    assert msg.text == "hello"


def test_message_assistant_from_str():
    msg = Message.assistant("hi there")
    assert msg.role == "assistant"
    assert msg.text == "hi there"


def test_message_from_list_of_parts():
    parts = [TextPart(text="a"), TextPart(text="b")]
    msg = Message.assistant(parts)
    assert msg.role == "assistant"
    assert msg.text == "ab"


def test_message_text_property_ignores_non_text_parts():
    msg = Message(
        role="assistant",
        parts=[
            TextPart(text="a"),
            ToolCallPart(id="1", name="f", arguments={}),
            TextPart(text="b"),
        ],
    )
    assert msg.text == "ab"


def test_message_round_trip_from_dict():
    data = {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
    msg = Message.model_validate(data)
    assert msg.role == "user"
    assert msg.text == "hi"
    assert msg.model_dump(mode="json") == data


def test_message_is_frozen():
    msg = Message.user("hi")
    with pytest.raises(ValidationError):
        msg.role = "assistant"


# --- Usage ---


def test_usage_default_is_zero_cost_none_complete_true():
    u = Usage()
    assert u.input_tokens == 0
    assert u.output_tokens == 0
    assert u.cost_usd is None
    assert u.cost_complete is True


def test_usage_add_sums_token_fields():
    a = Usage(
        input_tokens=10,
        output_tokens=5,
        cache_read_tokens=1,
        cache_creation_tokens=2,
        reasoning_tokens=3,
    )
    b = Usage(
        input_tokens=1,
        output_tokens=2,
        cache_read_tokens=3,
        cache_creation_tokens=4,
        reasoning_tokens=5,
    )
    c = a + b
    assert c.input_tokens == 11
    assert c.output_tokens == 7
    assert c.cache_read_tokens == 4
    assert c.cache_creation_tokens == 6
    assert c.reasoning_tokens == 8


def test_usage_add_both_costs_known_and_complete():
    a = Usage(cost_usd=1.0, cost_complete=True)
    b = Usage(cost_usd=2.5, cost_complete=True)
    c = a + b
    assert c.cost_usd == 3.5
    assert c.cost_complete is True


def test_usage_add_both_none_both_complete_stays_none_and_complete():
    # Two default (zero-usage) Usages: nothing happened on either side.
    a = Usage()
    b = Usage()
    c = a + b
    assert c.cost_usd is None
    assert c.cost_complete is True


def test_usage_add_default_none_plus_known_cost():
    a = Usage()  # zero-usage default
    b = Usage(cost_usd=2.0, cost_complete=True)
    c = a + b
    assert c.cost_usd == 2.0
    assert c.cost_complete is True


def test_usage_add_unknown_cost_plus_known_cost_marks_incomplete():
    a = Usage(cost_usd=None, cost_complete=False)  # adapter: pricing unknown
    b = Usage(cost_usd=2.0, cost_complete=True)
    c = a + b
    assert c.cost_usd == 2.0
    assert c.cost_complete is False


def test_usage_add_both_none_one_incomplete_stays_none_and_incomplete():
    a = Usage(cost_usd=None, cost_complete=False)
    b = Usage(cost_usd=None, cost_complete=True)
    c = a + b
    assert c.cost_usd is None
    assert c.cost_complete is False


def test_usage_add_both_known_but_one_incomplete_still_sums_and_flags_incomplete():
    a = Usage(cost_usd=1.0, cost_complete=False)
    b = Usage(cost_usd=1.0, cost_complete=True)
    c = a + b
    assert c.cost_usd == 2.0
    assert c.cost_complete is False


def test_usage_is_frozen():
    u = Usage()
    with pytest.raises(ValidationError):
        u.cost_usd = 5.0


# --- StopReason ---


def test_stop_reason_values_and_str_compat():
    assert StopReason.END_TURN == "end_turn"
    assert StopReason.MAX_TOKENS == "max_tokens"
    assert StopReason.TOOL_USE == "tool_use"
    assert StopReason.REFUSAL == "refusal"
    assert StopReason.ERROR == "error"
    assert StopReason.OTHER == "other"
    assert isinstance(StopReason.END_TURN, str)
