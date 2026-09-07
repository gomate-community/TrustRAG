from unittest.mock import patch

from trustrag.modules.generator.chat import AtlasCloudChat


@patch("trustrag.modules.generator.chat.OpenAI")
def test_atlas_cloud_chat_defaults(mock_openai):
    chat = AtlasCloudChat(key="test-key")

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://api.atlascloud.ai/v1",
    )
    assert chat.model_name == "openai/gpt-4.1-mini"


@patch("trustrag.modules.generator.chat.OpenAI")
def test_atlas_cloud_chat_accepts_overrides(mock_openai):
    chat = AtlasCloudChat(
        key="test-key",
        model_name="qwen/qwen3.5-flash",
        base_url="https://example.com/v1",
    )

    mock_openai.assert_called_once_with(
        api_key="test-key",
        base_url="https://example.com/v1",
    )
    assert chat.model_name == "qwen/qwen3.5-flash"
