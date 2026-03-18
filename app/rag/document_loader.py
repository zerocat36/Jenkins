from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

def load_paper(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])


# 1단계: 텍스트 추출
paper_text = load_paper("sample path")

# 2단계: 문서 분할
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=150)
docs = splitter.create_documents([paper_text])

# 3단계: 벡터 저장 (전역 변수에 저장)
global VECTORSTORE
VECTORSTORE = FAISS.from_documents(docs, OpenAIEmbeddings())