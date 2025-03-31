# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import FAISS
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

    def __init__(self, llm_model: str = "llama2:7b"):
        """
        Initialize the ChatPDF instance with Llama 2 7B model.
        """
        # Use llama2 for both chat and embeddings
        self.model = ChatOllama(
            model=llm_model,
            temperature=0.1,
            num_ctx=4096  # Increased context window
        )
        
        self.embeddings = OllamaEmbeddings(
            model=llm_model,
            num_ctx=4096  # Increased context window
        )
        
        # Optimize chunk size
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
        try:
            logger.info(f"Starting ingestion for file: {pdf_file_path}")
            
            # Load PDF and log its content
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            logger.info(f"Loaded {len(docs)} pages from PDF")
            for i, doc in enumerate(docs):
                logger.info(f"Page {i+1} preview: {doc.page_content[:100]}...")
            
            # Split into chunks and log
            chunks = self.text_splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                logger.info(f"Chunk {i+1} preview: {chunk.page_content[:100]}...")
            
            chunks = filter_complex_metadata(chunks)
            logger.info(f"Filtered chunks count: {len(chunks)}")

            # Create vector store and log
            self.vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            logger.info(f"Created vector store with {len(chunks)} embeddings")
            
            # Test vector store
            test_query = "test query"
            test_results = self.vector_store.similarity_search(test_query, k=1)
            logger.info(f"Vector store test search successful. Found {len(test_results)} results.")
            
        except Exception as e:
            logger.error(f"Error during PDF ingestion: {str(e)}")
            raise

    def ask(self, query: str, k: int = 5, score_threshold: float = 0.1):
        """
        Answer a query using the RAG pipeline.
        
        Args:
            query: The user's question
            k: Number of documents to retrieve
            score_threshold: Minimum similarity score for retrieved documents (default: 0.1)
        """
        if not self.vector_store:
            logger.error("No vector store found")
            raise ValueError("No vector store found. Please ingest a document first.")

        try:
            # First try with score threshold
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "k": k,
                    "score_threshold": score_threshold
                }
            )
            logger.info(f"Created retriever with k={k} and score_threshold={score_threshold}")

            # Log the query
            logger.info(f"Processing query: {query}")
            
            # Get documents and log them
            retrieved_docs = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # If no documents found with score threshold, try without it
            if not retrieved_docs:
                logger.info("No documents found with score threshold, trying without threshold")
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": k}
                )
                retrieved_docs = self.retriever.invoke(query)
                logger.info(f"Retrieved {len(retrieved_docs)} documents without threshold")

            if not retrieved_docs:
                logger.warning("No relevant documents found for the query")
                return "I could not find any relevant information in the document to answer your question. Please try rephrasing your question."

            # Log each retrieved document with its score if available
            for i, doc in enumerate(retrieved_docs):
                logger.info(f"Retrieved doc {i+1}:")
                logger.info(f"Content preview: {doc.page_content[:200]}...")
                if hasattr(doc, 'metadata'):
                    logger.info(f"Metadata: {doc.metadata}")

            formatted_input = {
                "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
                "question": query,
            }

            # Build and run the chain
            chain = (
                RunnablePassthrough()
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Generating response using the LLM")
            response = chain.invoke(formatted_input)
            logger.info(f"Generated response: {response[:200]}...")
            return response

        except Exception as e:
            logger.error(f"Error during retrieval or response generation: {str(e)}")
            logger.exception("Full traceback:")
            return f"An error occurred while processing your question: {str(e)}"

    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
