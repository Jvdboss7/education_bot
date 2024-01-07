from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import faiss
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import *

def faiss_vector_db():
    # loading the data
    dir_loader = DirectoryLoader(DATA_DIR_PATH,
                                 glob='*.pdf',
                                 loader_cls=PyPDFLoader)
    docs = dir_loader.load()
    print("PDF's Loaded")

    # creating the split in the data
    txt_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
                                                  chunk_overlap=CHUNK_OVERLAP)
    inp_text = txt_splitter.split_documents(docs)
    print("Data Chunks Created")

    # Create the embeddings from hugging face
    hfembeddings = HuggingFaceEmbeddings(model_name=EMBEDDER,
                                         model_kwargs = MODEL_KWARGS)
    db = faiss.FAISS.from_documents(inp_text,hfembeddings)
    db.save_local(VECTOR_DB_PATH)
    print("Vector Store Creation Completed")
if __name__=="__main__":
    faiss_vector_db()