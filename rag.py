# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "llama2:7b", embedding_model: str = "llama2:7b"):
        """
        Initialize the ChatPDF instance with Llama 2 7B model.
        """
        # Use llama2:7b for both chat and embeddings
        self.model = ChatOllama(
            model=llm_model,
            temperature=0.1,
            num_ctx=4096  # Increased context window
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model,
            num_ctx=4096  # Increased context window
        )
        # Optimize chunk size for llama2 model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=200,
            length_function=len
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            
            Here is the relevant context from the document:
            ---
            {context}
            ---
            
            Question: {question}
            
            Instructions:
            1. Base your answer ONLY on the context provided above
            2. If you can't find the answer in the context, say "I cannot find this information in the document"
            3. Keep your answer clear and direct
            4. Use bullet points if listing multiple items
            5. If the context contains relevant information, always provide an answer
            
            Answer:
            """
        )
        self.vector_store = None
        self.retriever = None

    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        logger.info(f"Loaded {len(docs)} pages from PDF")
        
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")
        chunks = filter_complex_metadata(chunks)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="chroma_db",
        )
        logger.info(f"Created vector store with {len(chunks)} embeddings")

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            raise ValueError("No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="mmr",  # Use MMR for better diversity in results
                search_kwargs={
                    "k": k,
                    "fetch_k": k * 2,  # Fetch more candidates for MMR to choose from
                    "lambda_mult": 0.7  # Balance between relevance and diversity
                }
            )

        logger.info(f"Retrieving context for query: {query}")
        try:
            retrieved_docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            if not retrieved_docs:
                return "I could not find any relevant information in the document to answer your question. Please try rephrasing your question."

            # Log retrieved content for debugging
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Chunk {i+1} content preview: {doc.page_content[:100]}...")

            formatted_input = {
                "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
                "question": query,
            }

            # Build the RAG chain
            chain = (
                RunnablePassthrough()
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Generating response using the LLM")
            response = chain.invoke(formatted_input)
            logger.info("Successfully generated response")
            return response

        except Exception as e:
            logger.error(f"Error during retrieval or response generation: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
