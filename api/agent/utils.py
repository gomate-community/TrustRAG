import asyncio
import platform
import sys
from typing import Dict, List, Any, Optional


def format_source_metadata(metadata: Dict[str, Any]) -> str:
    """
    Format source metadata into a human-readable string.
    
    Args:
        metadata: Dictionary of metadata for a source
        
    Returns:
        Formatted string
    """
    if not metadata:
        return "No metadata available"
    
    parts = []
    
    # Add title if available
    if "title" in metadata and metadata["title"]:
        parts.append(f"Title: {metadata['title']}")
    
    # Add author if available
    if "author" in metadata and metadata["author"]:
        parts.append(f"Author: {metadata['author']}")
    
    # Add publication date if available
    if "date" in metadata and metadata["date"]:
        parts.append(f"Date: {metadata['date']}")
    
    # Add URL if available
    if "url" in metadata and metadata["url"]:
        parts.append(f"URL: {metadata['url']}")
    
    # Add any other metadata fields
    for key, value in metadata.items():
        if key not in ["title", "author", "date", "url"] and value:
            # Format the key with proper capitalization
            formatted_key = " ".join(word.capitalize() for word in key.split("_"))
            parts.append(f"{formatted_key}: {value}")
    
    return "\n".join(parts)


def add_event_loop_policy():
    """Add event loop policy for Windows if needed."""
    if platform.system() == "Windows":
        try:
            # Set event loop policy for Windows
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception as e:
            print(f"Error setting event loop policy: {e}")


def chunk_text(text: str, chunk_size: int = 4000) -> List[str]:
    """
    Split text into chunks of specified size.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    current_chunk = ""
    
    # Try to split on sentences or paragraphs
    paragraphs = text.split("\n\n")
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= chunk_size:
            if current_chunk:
                current_chunk += "\n\n"
            current_chunk += paragraph
        else:
            # If the paragraph itself is larger than chunk_size, split by sentences
            if len(paragraph) > chunk_size:
                sentences = paragraph.split(". ")
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                        if current_chunk:
                            current_chunk += ". "
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If the sentence itself is too large, split by characters
                        if len(sentence) > chunk_size:
                            sentence_chunks = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size)]
                            chunks.extend(sentence_chunks[:-1])
                            current_chunk = sentence_chunks[-1]
                        else:
                            current_chunk = sentence
            else:
                chunks.append(current_chunk)
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks 