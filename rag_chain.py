# crag.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import GooglePalmEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

def load_model(model_name):
    return ChatGoogleGenerativeAI(model=model_name)

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    embeddings = GooglePalmEmbeddings()
    vectorstore = Chroma.from_texts(chunks, embeddings)
    
    return vectorstore.as_retriever()

def grade_documents(question, documents):
    llm = load_model("gemini-pro")
    
    system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question} \n\nIs this document relevant to the question? Answer with 'yes' or 'no'."),
    ])
    
    grading_chain = grade_prompt | llm | StrOutputParser()
    
    relevant_docs = []
    for doc in documents:
        grade = grading_chain.invoke({"question": question, "document": doc.page_content})
        if "yes" in grade.lower():
            relevant_docs.append(doc)
    
    return relevant_docs

def generate_answer(question, documents):
    llm = load_model("gemini-pro")
    
    prompt = ChatPromptTemplate.from_messages([
    ("human", "Use the following context to answer the question.\n\nContext: {context}\n\nQuestion: {question}")
    ])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=process_text("\n\n".join([doc.page_content for doc in documents])),
        chain_type_kwargs={"prompt": prompt}
    )
    
    return chain.run(question)

def crag_pipeline(question, all_text):
    retriever = process_text(all_text)
    initial_docs = retriever.get_relevant_documents(question)
    
    relevant_docs = grade_documents(question, initial_docs)
    
    if not relevant_docs:
        # If no relevant documents, you might want to implement a query reformulation step here
        # For simplicity, we'll just use web search as a fallback
        from langchain_community.tools.tavily_search import TavilySearchResults
        web_search = TavilySearchResults()
        web_results = web_search.run(question)
        return f"No relevant documents found. Web search result: {web_results}"
    
    answer = generate_answer(question, relevant_docs)
    return answer