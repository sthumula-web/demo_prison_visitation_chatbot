import os 
import sys 
import shutil
import re 

sys.path.append('../..')

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

os.environ["USER_AGENT"] = "PA-DOC-Chatbot/1.0"


pdf_files = [
    "Visitor-Guide (1).pdf",
    "2024 Friends and Family Handbook.pdf",
    "Professional-Visitor-Guide.pdf",
    "Call-Guide_Android.pdf",
    "Call-Guide_iOS.pdf",
    "Call-Guide_Win-Mac.pdf",
]

# ── add new URLs here ────────────────────────────────────────────
urls = [
    "https://www.pa.gov/agencies/cor/frequently-asked-questions",
    "https://www.pa.gov/services/cor/access-the-doc-inmate-visitation-system",
    "https://www.pa.gov/agencies/cor/find-or-contact-inmate/transportation-services",
    "https://www.pa.gov/agencies/cor/inmates-and-reentrants",
    "https://www.pa.gov/services/cor/request-to-perform-in-prison",
    "https://www.pa.gov/services/cor/apply-to-volunteer-in-a-prison",
    "https://www.pa.gov/agencies/cor/find-or-contact-inmate/mail-rules#general",
    "https://www.pa.gov/services/cor/send-mail",
    "https://www.pa.gov/agencies/cor/find-or-contact-inmate/commissary-lists",
    "https://www.pa.gov/services/cor/contact-an-inmate",
    "https://www.pa.gov/services/cor/custody-status-updates",
    "https://www.pa.gov/agencies/cor/inmates-and-reentrants/pennsylvania-correctional-industries",
    "https://www.pa.gov/agencies/cor/find-or-contact-inmate/inmate-visitation",
    "https://www.pa.gov/services/cor/locate-an-inmate-or-parolee",
]

pdf_docs = [] 
for pdf in pdf_files: 
    docs = PyPDFLoader(pdf).load()
    pdf_docs.extend(docs)
    

def clean_web_content(docs):
    for doc in docs:
        text = doc.page_content
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        doc.page_content = text.strip()
    return docs


web_docs = []
for url in urls:
    docs = WebBaseLoader(url).load()
    docs = clean_web_content(docs)
    web_docs.extend(docs)
    
# ── split ────────────────────────────────────────────────────────
pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", " ", ""]
 )

web_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
 )

split_pdf_docs = pdf_splitter.split_documents(pdf_docs)
split_web_docs = web_splitter.split_documents(web_docs)

all_docs = split_pdf_docs + split_web_docs

# embedding from hugging face
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = 'docs/chroma/'
if os.path.exists(persist_directory):
    shutil.rmtree(persist_directory)
    print("Old Chroma index deleted, rebuilding fresh...")

vectordb = Chroma.from_documents(
    documents=all_docs,
    embedding=embedding, 
    persist_directory=persist_directory
)
print(f"Done! New index has {vectordb._collection.count()} chunks")




