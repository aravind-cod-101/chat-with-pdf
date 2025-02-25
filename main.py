import os
import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader,PyMuPDFLoader
from langchain_ollama import OllamaEmbeddings,ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from langchain.schema import Document
import easyocr
model = 'llama3.2'
embedding_model='nomic-embed-text'

# import nltk
# nltk.download('all')
# print(nltk.data.path)

# make the file_path as input arg
# data/eb4a463b-ea54-42d4-b8a9-48f9d38661ed.pdf


file_path = input('Enter the file path: ').strip()
# data = []
# Extract data from uploaded pdf
if file_path and os.path.exists(file_path):
        loader = UnstructuredPDFLoader(file_path=file_path,strategy='hi_res')
        data = loader.load()
    # for f in os.listdir(file_path):
    #     loader = UnstructuredPDFLoader(f'data/{f}',strategy='hi_res')
    #     data.extend(loader.load())
    # loader = PDFPlumber Loader(file_path)
    # loader = PyMuPDFLoader(file_path)
    # images = convert_from_path(file_path)
    # print(data)
    # data = [Document(page_content=pytesseract.image_to_string(img)) for img in images]
    # reader = easyocr.Reader(['en'])
    # text = []
    # for img in convert_from_path(file_path):
    #     ext_text = reader.readtext(img)
    #     page_text = "\n".join(ext_text)
    #     text.append(page_text)
    # full_text = "\n\n".join(text)
    # data = Document(page_content=full_text)
        print(data)
        print('File Load Completed...')
else:
    print("Upload a PDF file")

# Convert pdf data to chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200,chunk_overlap=300)
chunks = text_splitter.split_documents(data)
print('Done splitting...')

# Create embeddings and add to vector database
ollama.pull(embedding_model)
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OllamaEmbeddings(model=embedding_model),
    collection_name="document-chat-rag"
)
print('Done adding to vector database...')

# Retrieval
llm = ChatOllama(model=model)
QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are AI language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from vector database. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions seperated by newlines.
    Original question: {question}""",
)

retriever = MultiQueryRetriever.from_llm(
    vector_db.as_retriever(), llm=llm, prompt=QUERY_PROMPT
)
print("Done creating retriever...")

template = """Answer the question ONLY based on the following context:
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print('Done creating chain...')


print('******************Start Chatting******************')
while True:
    question = input('Enter question to ask: ').strip()
    if not question or question.lower() == 'exit':
        break
    print("=================Generating Response=================")
    res = chain.invoke(input=question)
    print(res)
    print('=====================================================')
print('+++++++++++++++++++++++The END+++++++++++++++++++++++')