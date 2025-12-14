"""
RAG Agent untuk pencarian lowongan berbasis semantic search
Menggunakan Qdrant Cloud + OpenAI Embeddings
"""

import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.http import models
import os
from dotenv import load_dotenv

# Update semua import LangChain seperti ini:
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class RAGAgent:
    """Agent untuk Retrieval-Augmented Generation dari job dataset"""
    
    def __init__(
        self, 
        collection_name: str = "indonesian_jobs",
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4o-mini"
    ):
        """
        Inisialisasi RAG Agent dengan Qdrant dan OpenAI
        
        Args:
            collection_name: Nama collection di Qdrant
            qdrant_url: URL Qdrant Cloud (default dari env)
            qdrant_api_key: API key Qdrant (default dari env)
            embedding_model: Model embeddings OpenAI
            llm_model: Model LLM untuk generation
        """
        # Setup Qdrant Client
        self.collection_name = collection_name
        self.client = self._init_qdrant_client(qdrant_url, qdrant_api_key)
        
        # Setup Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Setup LLM untuk generation
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,  # Deterministic untuk accuracy
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prompt template dengan strict grounding
        self.prompt = ChatPromptTemplate.from_template("""
        Anda adalah Career Assistant yang membantu mencari lowongan pekerjaan.
        
        **ATURAN KETAT:**
        1. HANYA gunakan informasi dari context di bawah
        2. JANGAN membuat informasi yang tidak ada di context
        3. Jika tidak tahu, katakan "Berdasarkan data yang ada, saya tidak menemukan informasi tersebut"
        
        **CONTEXT (Data Lowongan):**
        {context}
        
        **PERTANYAAN USER:**
        {question}
        
        **JAWABAN ANDA:**
        """)
    
    def _init_qdrant_client(self, url: Optional[str], api_key: Optional[str]) -> QdrantClient:
        """Initialize Qdrant client dengan fallback ke environment variables"""
        try:
            qdrant_url = url or os.getenv("QDRANT_URL")
            qdrant_key = api_key or os.getenv("QDRANT_API_KEY")
            
            if not qdrant_url:
                raise ValueError("Qdrant URL tidak ditemukan. Set QDRANT_URL di .env")
            
            return QdrantClient(
                url=qdrant_url,
                api_key=qdrant_key,
                timeout=30  # Timeout 30 detik
            )
        except Exception as e:
            logger.error(f"Gagal koneksi ke Qdrant: {str(e)}")
            raise
    
    def retrieve_documents(self, query: str, limit: int = 5) -> List[Document]:
        """
        Retrieve dokumen relevan dari Qdrant berdasarkan semantic similarity
        
        Args:
            query: Query pencarian (natural language)
            limit: Jumlah hasil maksimal
            
        Returns:
            List of Documents dengan metadata
        """
        try:
            # 1. Generate embedding dari query
            query_vector = self.embeddings.embed_query(query)
            
            # 2. Search di Qdrant dengan filter untuk dataset Indonesia
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="country",
                            match=models.MatchValue(value="Indonesia")
                        )
                    ]
                ),
                limit=limit,
                with_payload=True
            )
            
            # 3. Format results menjadi LangChain Documents
            documents = []
            for hit in search_result:
                # Extract informasi dari payload
                payload = hit.payload or {}
                
                # Format konten yang informatif
                content_parts = []
                if "title" in payload:
                    content_parts.append(f"Posisi: {payload['title']}")
                if "company" in payload:
                    content_parts.append(f"Perusahaan: {payload['company']}")
                if "description" in payload:
                    # Truncate description jika terlalu panjang
                    desc = payload['description'][:500] + "..." if len(payload['description']) > 500 else payload['description']
                    content_parts.append(f"Deskripsi: {desc}")
                if "requirements" in payload:
                    content_parts.append(f"Persyaratan: {payload['requirements']}")
                if "salary_range" in payload:
                    content_parts.append(f"Estimasi Gaji: {payload['salary_range']}")
                if "location" in payload:
                    content_parts.append(f"Lokasi: {payload['location']}")
                if "work_type" in payload:
                    content_parts.append(f"Tipe Kerja: {payload['work_type']}")
                
                page_content = "\n".join(content_parts)
                
                # Simpan metadata untuk referensi
                metadata = {
                    "id": str(hit.id),
                    "score": float(hit.score),
                    "source": "qdrant_job_dataset",
                    **{k: str(v) for k, v in payload.items() if k != 'description'}
                }
                
                documents.append(Document(
                    page_content=page_content,
                    metadata=metadata
                ))
            
            logger.info(f"Retrieved {len(documents)} documents for query: '{query}'")
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def run(self, query: str) -> str:
        """
        End-to-end RAG pipeline: Retrieve -> Augment -> Generate
        
        Args:
            query: Pertanyaan user tentang lowongan
            
        Returns:
            Jawaban berbasis data
        """
        # Step 1: Retrieval
        documents = self.retrieve_documents(query, limit=3)
        
        if not documents:
            return (
                "Maaf, saya tidak menemukan lowongan yang sesuai dengan kriteria Anda. "
                "Coba gunakan kata kunci yang lebih spesifik atau perjelas bidang yang Anda cari."
            )
        
        # Step 2: Augmentation - Prepare context
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[LOWONGAN {i}]")
            context_parts.append(doc.page_content)
            if "salary_range" in doc.metadata:
                context_parts.append(f"Estimasi Gaji: {doc.metadata['salary_range']}")
            context_parts.append("")  # Blank line separator
        
        context_text = "\n".join(context_parts)
        
        # Step 3: Generation dengan RAG chain
        try:
            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "context": context_text,
                "question": query
            })
            
            # Tambahkan disclaimer
            response += "\n\n📌 *Catatan: Informasi berdasarkan data lowongan yang tersedia. Hubungi perusahaan langsung untuk detail lengkap.*"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Maaf, terjadi kesalahan dalam memproses permintaan Anda. Silakan coba lagi."

# ============================================================================
# PSEUDOCODE ASLI (dijadikan komentar untuk referensi)
# ============================================================================
"""
PSEUDOCODE - RAG AGENT (Original Concept)

CLASS RAGAgent:
    FUNCTION __init__(qdrant_url, qdrant_key, llm_model):
        SET self.client = CONNECT to Qdrant Cloud
        SET self.embeddings = INIT OpenAIEmbeddings()
        SET self.llm = llm_model

    FUNCTION retrieve_documents(query_text):
        # Step 1: Embed query user jadi angka
        query_vector = self.embeddings.embed_query(query_text)

        # Step 2: Search di Qdrant
        search_hits = self.client.search(
            collection_name="indonesian_jobs",
            query_vector=query_vector,
            limit=5
        )
        RETURN search_hits (list of job descriptions)

    FUNCTION run(user_query):
        # Step 1: Ambil data relevan
        retrieved_jobs = self.retrieve_documents(user_query)
        
        # Step 2: Gabungkan jadi Context string
        context_str = ""
        FOR job IN retrieved_jobs:
            context_str += job.payload['title'] + ": " + job.payload['description'] + "\n"

        # Step 3: Generate Jawaban
        prompt = f'''
        Role: Career Assistant.
        Context: {context_str}
        User Question: {user_query}
        Task: Jawab pertanyaan user hanya berdasarkan Context di atas.
        '''
        
        answer = self.llm.predict(prompt)
        RETURN answer
"""
