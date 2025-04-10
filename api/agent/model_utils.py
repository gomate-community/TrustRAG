"""
Utility functions for interacting with language models.
"""

import json
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
from loguru import logger

from api.agent.providers import get_model


async def generate_completion(
        prompt: str,
        system_message: str = "You are a helpful AI research assistant.",
        temperature: float = 0.7,
        json_response: bool = False,
        stream: bool = False
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generate a completion from the language model.
    
    Args:
        prompt: The prompt to send to the model
        system_message: The system message to use
        temperature: The temperature to use for generation
        json_response: Whether to request a JSON response
        stream: Whether to stream the response
        
    Returns:
        The model's response as a string or an async generator of response chunks
    """
    model_config = get_model()

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]

    # Prepare the request parameters
    request_params = {
        "model": model_config["model"],
        "messages": messages,
        "temperature": temperature,
    }

    # Add response format if JSON is requested
    if json_response:
        request_params["response_format"] = {"type": "json_object"}

    # Add streaming parameter if requested
    if stream:
        request_params["stream"] = True
    # logger.debug(request_params)
    try:
        # Make the API call
        response = await model_config["async_client"].chat.completions.create(**request_params)

        if stream:
            # Return an async generator for streaming responses
            async def response_generator():
                collected_chunks = []
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        collected_chunks.append(content)
                        yield content

                # If no chunks were yielded, yield the empty string
                if not collected_chunks:
                    yield ""

            return response_generator()
        else:
            # Return the full response for non-streaming
            res = response.choices[0].message.content.split("</think>")[-1].strip()
            logger.debug(f"prompt: {prompt}\n\nGenerated completion: {res}")
            return res

    except Exception as e:
        logger.error(f"Error generating completion: {str(e)}")
        if stream:
            # Return an empty generator for streaming
            async def error_generator():
                yield f"Error: {str(e)}"

            return error_generator()
        else:
            # Return an error message for non-streaming
            return f"Error: {str(e)}"


async def generate_json_completion(
        prompt: str,
        system_message: str = "You are a helpful AI research assistant.",
        temperature: float = 0.7
) -> Dict[str, Any]:
    """
    Generate a JSON completion from the language model.
    
    Args:
        prompt: The prompt to send to the model
        system_message: The system message to use
        temperature: The temperature to use for generation
        
    Returns:
        The model's response parsed as a JSON object
    """
    response_text = ""
    try:
        response_text = await generate_completion(
            prompt=prompt,
            system_message=system_message,
            temperature=temperature,
            json_response=True
        )

        # Parse the JSON response
        result = json.loads(response_text)
        logger.debug(f"prompt: {prompt}\n\nGenerated JSON completion: {result}")
        return result

    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {str(e)}")
        logger.error(f"Response text: {response_text}")
        return {}

    except Exception as e:
        logger.error(f"Error in generate_json_completion: {str(e)}")
        return {}
