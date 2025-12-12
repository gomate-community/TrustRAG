from langchain.text_splitter import CharacterTextSplitter
text = "This is the text I would like to ch up. It is the example text for this exercise."


text_splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=2,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

print(text_splitter.create_documents([text]))