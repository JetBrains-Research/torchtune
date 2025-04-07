# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset

from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.data._messages import OpenAIToMessages
from torchtune.modules.transforms import Transform
from torchtune.datasets._packed import PackedDataset
from torchtune.datasets._sft import SFTDataset
from torchtune.modules.transforms.tokenizers import ModelTokenizer


DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

DEFAULT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
)

# needed for system prompt generation
def reverse_convert_to_tool(oai_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reverses the conversion for OpenAI style tool names (e.g., 'my_func' -> 'my.func')."""
    original_functions = []
    if not isinstance(oai_tools, list):
        return original_functions # Return empty if input is not a list
    for tool in oai_tools:
        if isinstance(tool, dict) and tool.get("type") == "function" and "function" in tool:
            func = tool["function"]
            if isinstance(func, dict):
                name = func.get("name", "")
                # Only reverse the underscore substitution if no dot is already present.
                if isinstance(name, str) and "_" in name and "." not in name:
                    func["name"] = name.replace("_", ".")
                original_functions.append(func)
    return original_functions

class BfclMessageConverter(Transform):
    """
    Message transform for the BFCL dataset.

    This transform dynamically generates the system prompt for each sample based on the
    available tools, using the `reverse_convert_to_tool` logic to format function names.
    It then uses the standard `OpenAIToMessages` transform to convert the modified
    sample (with the generated system prompt) into the Torchtune Message format.

    Args:
        masking_strategy (str): Masking strategy to pass to the internal OpenAIToMessages.
            Defaults to "train_on_assistant".
    """
    def __init__(self, masking_strategy: str = "train_on_assistant"):
        # Internal transform to convert OpenAI messages after system prompt preparation
        self._openai_converter = OpenAIToMessages(masking_strategy=masking_strategy)


    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Prepares a single sample by generating the system prompt and then converting messages."""
        # Make a copy to avoid modifying the input dictionary directly, which can cause issues
        # with dataset caching or subsequent processing steps.
        prepared_sample = sample # .copy()

        tools = prepared_sample.get("tools", [])
        
        if len(tools) > 0:
            functions_converted = reverse_convert_to_tool(tools)
            # Use separators for compact JSON representation, matching the reference code
            functions_json = json.dumps(functions_converted, separators=(", ", ": "))
            system_prompt_content = DEFAULT_SYSTEM_PROMPT.format(functions=functions_json)

        messages = prepared_sample.get("messages", [])

        # if the first message is not a system message, insert the system message
        if len(tools) > 0 and messages[0]['role'] != 'system':
            messages.insert(0, {"role": "system", "content": system_prompt_content})

        prepared_sample["messages"] = messages

        # Now convert the prepared sample using the standard OpenAI converter
        # This returns a dict like {"messages": [Message(), ...]}
        final_messages_dict = self._openai_converter(prepared_sample)
        
        return final_messages_dict


def bfcl_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "JetBrains-Research/BFCL-trajectories",
    subset: str,  # Mandatory: specify the model subset, e.g., "qwen7bFC"
    packed: bool = False,
    filter_fn: Optional[Callable] = None,
    split: str = "train_full",
    max_seq_len: Optional[int] = None,
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[SFTDataset, PackedDataset]:
    """
    Builds a dataset from the `JetBrains-Research/BFCL-trajectories <https://huggingface.co/datasets/JetBrains-Research/BFCL-trajectories>`_
    dataset.

    This function requires specifying a `subset` corresponding to one of the models available
    in the dataset (e.g., "qwen7bFC", "gpt4ominiFC"). It loads the data using the Hugging Face
    `datasets` library, dynamically generates a system prompt for each sample based on the
    `tools` field using a custom message transform, and applies the necessary transforms for SFT training.

    The dataset contains "messages" in the OpenAI format, which are converted into a list
    of `Message` objects. By default, the model is trained on assistant responses only.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): Path to the dataset repository on Hugging Face.
            Default is ``JetBrains-Research/BFCL-trajectories``.
        subset (str): The specific model subset to load from the dataset (e.g., "qwen7bFC"). This is mandatory.
        packed (bool): Whether or not to pack the dataset to ``max_seq_len`` prior to training. Default is False.
            If True, `max_seq_len` must be specified.
        filter_fn (Optional[Callable]): Callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        split (str): ``split`` argument for ``datasets.load_dataset``. Defines which split of the data to load
            (e.g., "train_full", "test_solved"). Default is "train_full".
        max_seq_len (Optional[int]): Maximum sequence length for the tokenizer. Required if ``packed=True``.
            If ``packed=False``, this is passed to the tokenizer, but packing is not performed.
        **load_dataset_kwargs (Dict[str, Any]): Additional keyword arguments to pass to ``load_dataset``.

    Returns:
        Union[SFTDataset, PackedDataset]: The constructed dataset, either as a standard SFTDataset or
        a PackedDataset if ``packed=True``.

    Raises:
        ValueError: If ``packed`` is True and ``max_seq_len`` is not provided.
        ValueError: If `subset` is not provided.

    Example:
        >>> # Load the solved training examples for the qwen7bFC model
        >>> bfcl_qwen_ds = bfcl_dataset(
        ...     tokenizer=my_tokenizer,
        ...     subset="qwen7bFC",
        ...     split="train_solved",
        ...     max_seq_len=2048
        ... )
        >>> # The dataset now has system prompts generated based on the 'tools' field
        >>> for batch in DataLoader(bfcl_qwen_ds, batch_size=1):
        ...     # Accessing the first sample's messages
        ...     # Note: Actual access depends on how SFTDataset stores/yields data
        ...     # This is illustrative. You'd typically use the tokenized batch directly.
        ...     # print(batch[0]['messages']) # Hypothetical access
        ...     print(f"Batch token count: {batch['tokens'].shape[1]}") # More realistic
        Batch token count: ...
    """
    if not subset:
        raise ValueError(
            "You must specify a 'subset' when loading the BFCL dataset (e.g., 'qwen7bFC')."
        )

    # Update tokenizer max_seq_len if provided
    if max_seq_len is not None:
        tokenizer.max_seq_len = max_seq_len

    # Use the custom transform to handle dynamic system prompt generation
    message_transform = BfclMessageConverter(masking_strategy="train_on_assistant")

    # Pass the original source info and the custom message transform to SFTDataset
    ds = SFTDataset(
        source=source,
        name=subset,
        split=split,
        message_transform=message_transform,
        model_transform=tokenizer,
        filter_fn=filter_fn,
        **load_dataset_kwargs,
    )

    if packed:
        if max_seq_len is None:
            raise ValueError(
                "PackedDataset requires max_seq_len to be specified."
            )
        return PackedDataset(ds, max_seq_len=max_seq_len, padding_idx=tokenizer.pad_id)
    return ds