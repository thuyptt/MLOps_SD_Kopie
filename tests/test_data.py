import os
import sys
import pytest
from transformers import CLIPTokenizer

# Add the path to the data directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
from make_dataset import tokenize_captions

def test_main():
    @pytest.fixture
    def tokenizer():
        """
        Fixture to load the tokenizer. This will be used in the tests.
        The tokenizer is loaded once and shared across tests to save time and resources.
        """
        return CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")


    def test_tokenize_captions_single_string(tokenizer):
        """
        Test the tokenize_captions function with a single string caption.

        This test checks if the function can correctly handle individual string captions.
        It verifies that the output shape matches the expected dimensions.
        """
        examples = {"caption": ["A middle-aged man with gray hair and glasses, in a dark suit and polka dot tie, posing confidently with arms crossed, blurred indoor background."]}
        input_ids = tokenize_captions(examples, tokenizer, "caption")
        assert input_ids.shape[0] == 1, "Expected one example in the output."
        assert input_ids.shape[1] == tokenizer.model_max_length, "Expected the tokenized caption to match the model's max length."

    def test_tokenize_captions_list_of_strings(tokenizer):
        """
        Test the tokenize_captions function with a list of string captions.

        Warning:
        This test assumes that the input is a list of string captions and that the tokenizer's model_max_length is correctly set.
        It verifies that the function can handle a list of captions by selecting one randomly.
        """
        examples = {"caption": [["This is a test caption.", "This is another caption."]]}
        input_ids = tokenize_captions(examples, tokenizer, "caption")
        assert input_ids.shape[0] == 1, "Expected one example in the output."
        assert input_ids.shape[1] == tokenizer.model_max_length, "Expected the tokenized caption to match the model's max length."

    def test_tokenize_captions_empty_string(tokenizer):
        """
        Test the tokenize_captions function with an empty string caption.
        """
        examples = {"caption": [""]}
        input_ids = tokenize_captions(examples, tokenizer, "caption")
        assert input_ids.shape[0] == 1, "Expected one example in the output."
        assert input_ids.shape[1] == tokenizer.model_max_length, "Expected the tokenized caption to match the model's max length."

    def test_tokenize_captions_invalid_type(tokenizer):
        """
        Test the tokenize_captions function with an invalid type for caption.

        This test ensures that the function raises a ValueError when the caption type is invalid.
        It verifies that the function correctly handles incorrect input types by raising an error.
        """
        examples = {"caption": [12345]}
        with pytest.raises(ValueError):
            tokenize_captions(examples, tokenizer, "caption")


    def test_preprocess_train(tokenizer):
        """
        Test the preprocess_train function to ensure it adds tokenized captions.
        """
        examples = {"caption": ["This is a test caption."]}
        caption_column = "caption"

        def preprocess_train(examples):
            examples["input_ids"] = tokenize_captions(examples, tokenizer, caption_column)
            return examples

        processed_examples = preprocess_train(examples)
        assert "input_ids" in processed_examples, "Expected 'input_ids' key in processed examples."
        assert processed_examples["input_ids"].shape[0] == 1, "Expected one example in the output."
        assert processed_examples["input_ids"].shape[1] == tokenizer.model_max_length, "Expected the tokenized caption to match the model's max length."



    # def test_data_loading():

pass
