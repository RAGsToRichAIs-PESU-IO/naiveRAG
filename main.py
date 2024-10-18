from dotenv import load_dotenv
load_dotenv()

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

parser = LlamaParse(
    result_type='markdown',
)

file_extractor = {".pdf":parser}
output_docs=SimpleDirectoryReader(input_files=['./data/Git Cheatsheet.pdf'], file_extractor=file_extractor)
docs = output_docs.load_data()
md_text = ""
for doc in docs:
    md_text += doc.text

with open('output.md', 'w') as file_handle:
    file_handle.write(md_text)

print("Markdown file created successfully")