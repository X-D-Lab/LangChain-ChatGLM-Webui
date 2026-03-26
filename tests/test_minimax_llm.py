"""Unit tests and integration tests for MiniMax LLM provider."""

import os
import re
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# Add parent directory to path so we can import minimax_llm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestChatMiniMaxLLMInit(unittest.TestCase):
    """Test ChatMiniMaxLLM initialization and configuration."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_init_with_default_model(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        self.assertEqual(llm.model_name, "MiniMax-M2.7")
        self.assertEqual(llm.api_base, "https://api.minimax.io/v1")
        self.assertEqual(llm.api_key, "test-key-123")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_init_with_custom_model(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM(model_name="MiniMax-M2.7-highspeed")
        self.assertEqual(llm.model_name, "MiniMax-M2.7-highspeed")

    @patch.dict(os.environ, {}, clear=True)
    def test_init_without_api_key_raises(self):
        # Remove MINIMAX_API_KEY if present
        os.environ.pop("MINIMAX_API_KEY", None)
        from minimax_llm import ChatMiniMaxLLM
        with self.assertRaises(ValueError) as ctx:
            ChatMiniMaxLLM()
        self.assertIn("MINIMAX_API_KEY", str(ctx.exception))

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_llm_type(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        self.assertEqual(llm._llm_type, "ChatMiniMaxLLM")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_identifying_params(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        params = llm._identifying_params
        self.assertEqual(params["model_name"], "MiniMax-M2.7")
        self.assertEqual(params["api_base"], "https://api.minimax.io/v1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_load_llm_noop(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        # load_llm should be a no-op for cloud models
        llm.load_llm()  # Should not raise

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_openai_client_initialized(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        mock_openai.assert_called_once_with(
            api_key="test-key-123",
            base_url="https://api.minimax.io/v1",
        )
        self.assertIsNotNone(llm.client)


class TestChatMiniMaxLLMCall(unittest.TestCase):
    """Test ChatMiniMaxLLM _call method."""

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_basic_call(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        llm.client.chat.completions.create.return_value = mock_response

        result = llm._call("Say hello")
        self.assertEqual(result, "Hello, world!")
        llm.client.chat.completions.create.assert_called_once()

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_temperature_clamping_low(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        llm.temperature = 0.0  # Should be clamped to 0.01

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("test")
        call_kwargs = llm.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 0.01)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_temperature_clamping_high(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        llm.temperature = 1.5  # Should be clamped to 1.0

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "result"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("test")
        call_kwargs = llm.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["temperature"], 1.0)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_think_tag_stripping(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = (
            "<think>Let me think about this...</think>\nActual answer here"
        )
        llm.client.chat.completions.create.return_value = mock_response

        result = llm._call("test")
        self.assertNotIn("<think>", result)
        self.assertEqual(result, "Actual answer here")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_stop_tokens(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello STOP extra text"
        llm.client.chat.completions.create.return_value = mock_response

        result = llm._call("test", stop=["STOP"])
        self.assertNotIn("extra text", result)

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_history_appended(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        llm.history = []

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response 1"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("Question 1")
        self.assertEqual(len(llm.history), 1)
        self.assertEqual(llm.history[0][1], "Response 1")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_history_included_in_messages(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        llm.history = [["previous question", "previous answer"]]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "new answer"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("new question")
        call_kwargs = llm.client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        self.assertEqual(len(messages), 3)  # 1 user + 1 assistant from history + 1 new user
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "previous question")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "previous answer")
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"], "new question")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_none_content_returns_empty_string(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        llm.client.chat.completions.create.return_value = mock_response

        result = llm._call("test")
        self.assertEqual(result, "")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_model_passed_to_api(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM(model_name="MiniMax-M2.7-highspeed")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "fast response"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("test")
        call_kwargs = llm.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["model"], "MiniMax-M2.7-highspeed")

    @patch.dict(os.environ, {"MINIMAX_API_KEY": "test-key-123"})
    @patch("minimax_llm.OpenAI")
    def test_max_tokens_passed(self, mock_openai):
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM()
        llm.max_token = 2048

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "response"
        llm.client.chat.completions.create.return_value = mock_response

        llm._call("test")
        call_kwargs = llm.client.chat.completions.create.call_args[1]
        self.assertEqual(call_kwargs["max_tokens"], 2048)


class TestConfigIntegration(unittest.TestCase):
    """Test MiniMax integration with config.py."""

    def test_minimax_in_llm_model_dict(self):
        # Import config and verify MiniMax models are present
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # Re-import to get fresh config
        import importlib
        # Avoid torch import issues by reading config directly
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.py"
        )
        with open(config_path) as f:
            content = f.read()
        self.assertIn('"minimax"', content)
        self.assertIn('"MiniMax-M2.7"', content)
        self.assertIn('"MiniMax-M2.7-highspeed"', content)

    def test_config_syntax_valid(self):
        """Verify config.py llm_model_dict is valid Python."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.py"
        )
        with open(config_path) as f:
            content = f.read()
        # Extract the llm_model_dict definition and verify it parses
        start = content.index("llm_model_dict = {")
        # Find matching closing brace
        brace_count = 0
        end = start
        for i, c in enumerate(content[start:], start):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        dict_str = content[start + len("llm_model_dict = "):][:end - start - len("llm_model_dict = ")]
        parsed = eval(dict_str)
        self.assertIn("minimax", parsed)
        self.assertEqual(parsed["minimax"]["MiniMax-M2.7"], "MiniMax-M2.7")


class TestAppIntegration(unittest.TestCase):
    """Test MiniMax integration in app.py model dispatch."""

    def test_minimax_branch_in_app(self):
        """Verify app.py contains MiniMax model dispatch."""
        app_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "app.py"
        )
        with open(app_path) as f:
            content = f.read()
        self.assertIn("'minimax' in large_language_model.lower()", content)
        self.assertIn("from minimax_llm import ChatMiniMaxLLM", content)

    def test_minimax_models_in_dropdown_list(self):
        """Verify MiniMax models appear in the model list."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.py"
        )
        with open(config_path) as f:
            content = f.read()
        # Parse the dict
        start = content.index("llm_model_dict = {")
        brace_count = 0
        end = start
        for i, c in enumerate(content[start:], start):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break
        dict_str = content[start + len("llm_model_dict = "):][:end - start - len("llm_model_dict = ")]
        llm_model_dict = eval(dict_str)

        # Build model list the same way app.py does
        llm_model_list = []
        for i in llm_model_dict:
            for j in llm_model_dict[i]:
                llm_model_list.append(j)

        self.assertIn("MiniMax-M2.7", llm_model_list)
        self.assertIn("MiniMax-M2.7-highspeed", llm_model_list)


class TestReadmeIntegration(unittest.TestCase):
    """Test README includes MiniMax documentation."""

    def test_minimax_in_readme(self):
        readme_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "README.md"
        )
        with open(readme_path) as f:
            content = f.read()
        self.assertIn("MiniMax-M2.7", content)
        self.assertIn("MINIMAX_API_KEY", content)
        self.assertIn("platform.minimaxi.com", content)


class TestThinkTagRegex(unittest.TestCase):
    """Test think tag stripping regex patterns."""

    def test_simple_think_tag(self):
        text = "<think>thinking...</think>\nAnswer"
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        self.assertEqual(result, "Answer")

    def test_multiline_think_tag(self):
        text = "<think>\nline1\nline2\n</think>\n\nAnswer"
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        self.assertEqual(result, "Answer")

    def test_no_think_tag(self):
        text = "Just a normal response"
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        self.assertEqual(result, "Just a normal response")

    def test_empty_think_tag(self):
        text = "<think></think>Answer"
        result = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        self.assertEqual(result, "Answer")


# ===================== Integration Tests =====================

class TestMiniMaxIntegration(unittest.TestCase):
    """Integration tests that call the real MiniMax API.

    These tests require MINIMAX_API_KEY to be set in the environment.
    Skip if not available.
    """

    @classmethod
    def setUpClass(cls):
        api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not api_key:
            # Try loading from .env.local
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                ".env.local"
            )
            if os.path.exists(env_path):
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("MINIMAX_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            os.environ["MINIMAX_API_KEY"] = api_key
                            break
        if not api_key:
            raise unittest.SkipTest("MINIMAX_API_KEY not set")

    def test_basic_completion(self):
        """Test basic chat completion with MiniMax API."""
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM(model_name="MiniMax-M2.7-highspeed")
        llm.temperature = 0.01
        llm.max_token = 100
        result = llm._call("What is 2+2? Answer with just the number.")
        self.assertIn("4", result)

    def test_highspeed_model(self):
        """Test MiniMax-M2.7-highspeed model."""
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM(model_name="MiniMax-M2.7-highspeed")
        llm.temperature = 0.01
        llm.max_token = 100
        result = llm._call("What is the capital of France? Answer with just the city name.")
        self.assertIn("Paris", result)

    def test_chinese_response(self):
        """Test Chinese language Q&A (matching project's primary use case)."""
        from minimax_llm import ChatMiniMaxLLM
        llm = ChatMiniMaxLLM(model_name="MiniMax-M2.7-highspeed")
        llm.temperature = 0.01
        llm.max_token = 200
        result = llm._call("请用一句话回答：中国的首都是哪里？")
        self.assertIn("北京", result)


if __name__ == "__main__":
    unittest.main()
