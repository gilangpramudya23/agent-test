"""
Orchestrator Agent sebagai central router untuk semua agent
Mengklasifikasikan intent dan mendelegasikan ke agent yang tepat
"""

import logging
import re
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

logger = logging.getLogger(__name__)

class Orchestrator:
    """Central router untuk multi-agent system"""
    
    def __init__(
        self,
        rag_agent,
        sql_agent,
        advisor_agent,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0
    ):
        """
        Inisialisasi Orchestrator dengan semua agent
        
        Args:
            rag_agent: Instance RAGAgent
            sql_agent: Instance SQLAgent
            advisor_agent: Instance AdvisorAgent
            llm_model: Model untuk intent classification
            temperature: Temperature untuk LLM
        """
        self.rag_agent = rag_agent
        self.sql_agent = sql_agent
        self.advisor_agent = advisor_agent
        
        # LLM untuk intent classification
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Prompt untuk intent classification
        self.intent_prompt = ChatPromptTemplate.from_template("""
        Anda adalah Router AI yang mengklasifikasikan pertanyaan user ke kategori yang tepat.
        
        **PERTANYAAN USER:**
        {query}
        
        **KATEGORI YANG TERSEDIA:**
        1. RAG_QUERY - Untuk pertanyaan tentang lowongan pekerjaan, deskripsi kerja, persyaratan, perusahaan, atau pencarian pekerjaan berdasarkan kriteria (natural language)
           Contoh: 
           - "Cari lowongan backend developer di Jakarta"
           - "Apa persyaratan untuk data scientist?"
           - "Perusahaan apa yang mencari product manager?"
        
        2. SQL_QUERY - Untuk pertanyaan yang membutuhkan analisis data, statistik, perbandingan, atau query numerik
           Contoh:
           - "Berapa rata-rata gaji software engineer?"
           - "Tampilkan 5 kota dengan gaji tertinggi"
           - "Berapa banyak lowongan remote vs hybrid?"
           - "Trend gaji untuk data analyst dalam 6 bulan terakhir"
        
        3. ADVISOR_QUERY - Untuk pertanyaan tentang konsultasi karir, pengembangan skill, atau saran profesional
           Contoh:
           - "Bagaimana cara meningkatkan skill Python?"
           - "Apa sertifikasi yang dibutuhkan untuk cloud engineer?"
           - "Saran karir untuk fresh graduate IT"
        
        **ATURAN:**
        - Jika query mengandung kata: "gaji", "rata-rata", "statistik", "trend", "perbandingan", "tertinggi", "terendah" → SQL_QUERY
        - Jika query tentang "lowongan", "pekerjaan", "cari kerja", "perusahaan", "posisi" → RAG_QUERY
        - Jika query tentang "karir", "saran", "konsultasi", "skill", "development", "fresh graduate" → ADVISOR_QUERY
        - Jika ragu, default ke RAG_QUERY
        
        **HANYA respon dengan salah satu dari: RAG_QUERY, SQL_QUERY, ADVISOR_QUERY**
        
        **JAWABAN:**
        """)
        
        # Keywords untuk fast classification (fallback jika LLM gagal)
        self.keyword_mapping = {
            "sql": ["gaji", "salary", "rata-rata", "statistik", "analisis", "trend", 
                   "perbandingan", "tertinggi", "terendah", "berapa banyak", "jumlah",
                   "persentase", "distribusi", "histori", "tahun", "bulan"],
            "rag": ["lowongan", "pekerjaan", "job", "cari kerja", "perusahaan", 
                   "posisi", "vacancy", "opening", "hire", "rekrutmen",
                   "deskripsi", "persyaratan", "kualifikasi", "lamaran", "apply"],
            "advisor": ["karir", "career", "saran", "advice", "konsultasi", "consult",
                       "skill", "keahlian", "development", "pengembangan", "fresh graduate",
                       "senior", "junior", "promosi", "resume", "cv", "interview"]
        }
    
    def classify_intent(self, query: str) -> str:
        """
        Klasifikasikan intent query menggunakan kombinasi rule-based dan LLM
        
        Args:
            query: Pertanyaan user
            
        Returns:
            Intent: RAG_QUERY, SQL_QUERY, atau ADVISOR_QUERY
        """
        # 1. Rule-based classification dengan keywords
        query_lower = query.lower()
        
        sql_score = sum(1 for keyword in self.keyword_mapping["sql"] if keyword in query_lower)
        rag_score = sum(1 for keyword in self.keyword_mapping["rag"] if keyword in query_lower)
        advisor_score = sum(1 for keyword in self.keyword_mapping["advisor"] if keyword in query_lower)
        
        # Jika ada keyword yang jelas, gunakan rule-based
        if sql_score > rag_score and sql_score > advisor_score and sql_score >= 2:
            return "SQL_QUERY"
        elif rag_score > sql_score and rag_score > advisor_score and rag_score >= 2:
            return "RAG_QUERY"
        elif advisor_score > sql_score and advisor_score > rag_score and advisor_score >= 2:
            return "ADVISOR_QUERY"
        
        # 2. Gunakan LLM untuk classification yang lebih nuanced
        try:
            chain = self.intent_prompt | self.llm | StrOutputParser()
            intent = chain.invoke({"query": query})
            
            # Clean and validate intent
            intent = intent.strip().upper()
            valid_intents = ["RAG_QUERY", "SQL_QUERY", "ADVISOR_QUENT"]
            
            if intent in valid_intents:
                return intent
            else:
                # Default ke RAG jika intent tidak valid
                logger.warning(f"Invalid intent from LLM: {intent}, defaulting to RAG_QUERY")
                return "RAG_QUERY"
                
        except Exception as e:
            logger.error(f"Error in LLM intent classification: {str(e)}")
            # Default ke RAG jika error
            return "RAG_QUERY"
    
    def route_query(self, query: str) -> str:
        """
        Route query ke agent yang sesuai berdasarkan intent classification
        
        Args:
            query: Pertanyaan user
            
        Returns:
            Response dari agent yang sesuai
        """
        try:
            logger.info(f"Routing query: '{query}'")
            
            # Classify intent
            intent = self.classify_intent(query)
            logger.info(f"Detected intent: {intent}")
            
            # Route ke agent yang sesuai
            if intent == "SQL_QUERY":
                logger.info("Delegating to SQL Agent...")
                response = self.sql_agent.run(query)
                response = f"📊 **Analisis Data:**\n\n{response}"
                
            elif intent == "ADVISOR_QUERY":
                logger.info("Delegating to Advisor Agent (general advice)...")
                # Untuk advisor query general (tanpa CV)
                response = self._handle_general_advice(query)
                
            else:  # RAG_QUERY (default)
                logger.info("Delegating to RAG Agent...")
                response = self.rag_agent.run(query)
                response = f"🔍 **Hasil Pencarian Lowongan:**\n\n{response}"
            
            # Tambahkan metadata response
            response += f"\n\n---\n*Kategori: {intent.replace('_', ' ')}*"
            
            return response
            
        except Exception as e:
            logger.error(f"Error routing query: {str(e)}")
            return (
                f"❌ Maaf, terjadi kesalahan dalam memproses permintaan Anda.\n"
                f"Error: {str(e)[:100]}\n\n"
                f"Silakan coba dengan pertanyaan yang berbeda atau lebih spesifik."
            )
    
    def _handle_general_advice(self, query: str) -> str:
        """Handle general career advice queries (tanpa CV upload)"""
        try:
            # Gunakan LLM langsung untuk general advice
            advice_prompt = ChatPromptTemplate.from_template("""
            Anda adalah Career Consultant profesional. Berikan saran karir untuk pertanyaan berikut.
            
            **PERTANYAAN:**
            {query}
            
            **PETUNJUK:**
            1. Berikan saran yang praktis dan actionable
            2. Sesuaikan dengan konteks pasar kerja Indonesia
            3. Jika relevan, sarankan skill yang perlu dipelajari
            4. Gunakan bahasa Indonesia yang jelas dan profesional
            5. Jika memungkinkan, berikan contoh konkret
            
            **JAWABAN ANDA:**
            """)
            
            chain = advice_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"query": query})
            
            return f"💼 **Konsultasi Karir:**\n\n{response}"
            
        except Exception as e:
            logger.error(f"Error in general advice: {str(e)}")
            return "Maaf, saya sedang tidak bisa memberikan saran karir. Silakan coba lagi nanti."
    
    def analyze_cv(self, pdf_path: str) -> str:
        """
        Route CV analysis ke Advisor Agent
        
        Args:
            pdf_path: Path ke file CV PDF
            
        Returns:
            Rekomendasi karir berdasarkan CV
        """
        try:
            logger.info(f"Starting CV analysis for: {pdf_path}")
            
            # Validasi file
            if not pdf_path or not pdf_path.endswith('.pdf'):
                return "❌ File harus dalam format PDF. Silakan upload file PDF yang valid."
            
            # Delegate ke Advisor Agent
            response = self.advisor_agent.analyze_and_recommend(pdf_path)
            
            return f"📄 **Analisis CV & Rekomendasi Karir:**\n\n{response}"
            
        except FileNotFoundError:
            return "❌ File tidak ditemukan. Pastikan file PDF yang diupload benar."
        except Exception as e:
            logger.error(f"Error in CV analysis: {str(e)}")
            return f"❌ Terjadi kesalahan dalam menganalisis CV: {str(e)[:200]}"

# ============================================================================
# PSEUDOCODE ASLI (dijadikan komentar untuk referensi)
# ============================================================================
"""
PSEUDOCODE - ORCHESTRATOR (Original Concept)

FUNCTION route_query(user_query):
    # STEP 1: Pikirkan (Klasifikasi)
    intent = AI_Classifier(user_query)  # "SQL" atau "RAG"
    
    # STEP 2: Delegasikan Tugas
    IF intent == "SQL":
        RETURN sql_agent.run(user_query)
    ELSE IF intent == "RAG":
        RETURN rag_agent.run(user_query)
    ELSE:
        # Fallback ke RAG
        RETURN rag_agent.run(user_query)
"""