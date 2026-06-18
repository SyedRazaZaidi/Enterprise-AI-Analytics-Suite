import os
import spacy
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_classic.chains import RetrievalQA

# ---------------------------------------------------------
# Step 1: Config & API Setup
# ---------------------------------------------------------
# Load our local spaCy model to handle entity extraction (NER)
nlp = spacy.load("en_core_web_sm")

# TODO: Paste your Hugging Face Access Token here
os.environ["HF_TOKEN"] = ""

print("Booting up the AI orchestrator and vector DB...")

# ---------------------------------------------------------
# Step 2: Build the RAG Context (Mock Business Data)
# ---------------------------------------------------------
# Creating a dummy report to test our context retrieval.
# In a real setup, this would parse historical PDF reports.
context_text = """
BUSINESS CONTEXT REPORT - 2026:
The West region experienced a 15% drop in furniture sales during Q3 due to a massive supply chain disruption at the Los Angeles port.
Technology sales in the East region spiked in Q4 because of the new corporate hardware upgrade initiative.
Central region office supplies remained stagnant due to a lack of marketing budget.
"""

with open("business_context.txt", "w") as f:
    f.write(context_text)

# ---------------------------------------------------------
# Step 3: Embeddings & ChromaDB Storage
# ---------------------------------------------------------
loader = TextLoader("business_context.txt")
documents = loader.load()

# Chop the text into smaller chunks for the vector store
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Using a lightweight, free sentence transformer for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Spin up the local Chroma database
vector_store = Chroma.from_documents(texts, embeddings, collection_name="superstore_reports")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ---------------------------------------------------------
# Step 4: The RAG Orchestration Pipeline
# ---------------------------------------------------------

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.1,
    max_new_tokens=500,
    return_full_text=False
)

# 2. Wrap the endpoint in a Chat interface to appease the API router
llm = ChatHuggingFace(llm=endpoint)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever()
)
def process_query(query: str):
    """
    Core routing function. Extracts entities via spaCy, then queries ChromaDB.
    """
    print(f"\nUser Query: '{query}'")
    
    # Run the query through spaCy to find named entities (locations, etc.)
    doc = nlp(query)
    entities = {ent.label_: ent.text for ent in doc.ents}
    print(f"Entities Extracted: {entities}")
    
    # Query the RAG chain for the actual answer
    print("Searching vector store for historical context...")
    response = qa_chain.invoke(query)
    
    return response['result']

# ---------------------------------------------------------
# Step 5: Execute Test
# ---------------------------------------------------------
if __name__ == "__main__":
    # Test a qualitative question to ensure RAG is working
    test_question = "Why did furniture sales drop in the West region?"
    final_answer = process_query(test_question)
    
    print(f"\nFinal AI Answer: {final_answer}")