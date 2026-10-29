from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
import io
from langchain_core.documents import Document
import traceback



embedding_model=OllamaEmbeddings(
    model="nomic-embed-text"
)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chroma_db=Chroma(
    collection_name="pdf_data",
    persist_directory="my_pdf_data",
    embedding_function=embedding_model,
)

def process_uploaded_doc(uploaded_pdf):

    reader=PdfReader(io.BytesIO(uploaded_pdf.getvalue()))


    docs=[Document(page_content=page.extract_text() or "",metadata={"source":uploaded_pdf.name,"page":i}) for i , page in enumerate(reader.pages)]


    chunks=splitter.split_documents(docs)

    if len(chroma_db.get(include=[])['ids'])==0 :
        chroma_db.add_documents(chunks)

def get_context(user_que):
    res=chroma_db.similarity_search(
        k=5,
        query=user_que
    )
    return res

    
    


    
