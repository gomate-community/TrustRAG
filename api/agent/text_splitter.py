class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=4000, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text):
        """Split text into chunks of specified size with specified overlap."""
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size

            # If we're not at the end of the text, try to find a good breaking point
            if end < len(text):
                # Try to find a newline to break on
                newline_pos = text.rfind('\n', start, end)
                if newline_pos > start:
                    end = newline_pos + 1
                else:
                    # Try to find a space to break on
                    space_pos = text.rfind(' ', start, end)
                    if space_pos > start:
                        end = space_pos + 1

            # Add the chunk
            chunks.append(text[start:end])

            # Move the start position, accounting for overlap
            start = end - self.chunk_overlap

        return chunks
