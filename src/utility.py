from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader
import io
from langchain_core.documents import Document
import traceback
import hashlib


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

def process_uploaded_doc(uploaded_pdf,hash_file):

    reader=PdfReader(io.BytesIO(uploaded_pdf.getvalue()))


    docs=[Document(page_content=page.extract_text() or "",metadata={"source":uploaded_pdf.name,"page":i,"file_hash":hash_file}) for i , page in enumerate(reader.pages)]


    chunks=splitter.split_documents(docs)
    ## generating deterministing Id for each chunks for id based filtering
    chunks_id=[hashlib.sha256(f"{hash_file}:{i}".encode()).hexdigest() for i in range(len(chunks))]
    if chunks:
        chroma_db.add_documents(
            documents=chunks,
            ids=chunks_id
        )

def get_context(user_que,hash_file):
    res=chroma_db.similarity_search(
        k=5,
        query=user_que,
        filter={"file_hash":hash_file}
    )
    print(len(res))
    for i, doc in enumerate(res):
        print(f"\n\n RESULT_{i}::-->")
        print(doc.page_content)
    return res

    
    


    
