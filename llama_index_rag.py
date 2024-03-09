import openai
import os

from dotenv import load_dotenv

# load the environment file
load_dotenv(".env")

# set OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# use SimpleDirectoryReader for data ingestion (i.e., loading the PDF of the research paper)
documents = SimpleDirectoryReader("data").load_data()

# create a vector store using OpenAI embeddings
index = VectorStoreIndex.from_documents(documents)

# create a 'query_engine'
query_engine = index.as_query_engine()

# run a query
res = query_engine.query(
    "How long did it take to train the machine translation model?"
)

print(res.response)