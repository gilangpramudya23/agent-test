"""
Advisor Agent: analisis CV dan rekomendasi karir sederhana

File ini menyediakan kelas `AdvisorAgent` yang mengekstrak teks dari PDF CV,
meminta ringkasan dan rekomendasi dari LLM, dan (opsional) mencari lowongan
yang cocok menggunakan `RAGAgent`.
"""

import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate  # ✅ FIXED
from langchain_core.output_parsers import StrOutputParser  # ✅ FIXED
from pypdf import PdfReader

logger = logging.getLogger(__name__)


class AdvisorAgent:
    """Simple advisor agent that analyzes a CV PDF and gives recommendations."""

    def __init__(self, rag_agent=None, llm_model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.rag_agent = rag_agent
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
            Anda adalah Career Consultant profesional.
            Berikut adalah teks CV pengguna:
            {cv_text}

            Berikan ringkasan singkat (2-3 kalimat), tiga rekomendasi posisi yang cocok,
            dan tiga saran perbaikan CV atau skill yang perlu ditingkatkan.

            Jawab dalam bahasa Indonesia dengan format:
            Ringkasan:\n- ...\n\nRekomendasi Posisi:\n- ...\n\nSaran:\n- ...
            """
        )

    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError("File CV tidak ditemukan")

        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)

        return "\n".join(text_parts)

    def analyze_and_recommend(self, pdf_path: str) -> str:
        """Extract CV, ask LLM for recommendations, and optionally query RAGAgent."""
        try:
            cv_text = self._extract_text_from_pdf(pdf_path)

            chain = self.prompt | self.llm | StrOutputParser()
            response = chain.invoke({"cv_text": cv_text})

            # Optionally, enrich with RAG results if available
            if self.rag_agent:
                # use top skill/position keywords to find jobs
                query = "; ".join(response.splitlines()[:2])
                jobs = self.rag_agent.retrieve_documents(query, limit=3)
                if jobs:
                    jobs_text = "\n\nLowongan terkait:\n" + "\n---\n".join(doc.page_content for doc in jobs)
                    response += jobs_text

            return response

        except Exception as e:
            logger.error(f"AdvisorAgent error: {e}")
            return "Maaf, terjadi kesalahan saat menganalisis CV. Silakan coba lagi." 

# ============================================================================
# PSEUDOCODE ASLI (dijadikan komentar untuk referensi)
# ============================================================================
"""
PSEUDOCODE - SQL AGENT (Original Concept)

CLASS SQLAgent:
    FUNCTION __init__(database_path, llm_model):
        SET self.db = CONNECT to SQLite database at database_path
        SET self.llm = llm_model

    FUNCTION get_schema_info():
        RETURN string berisi "Table: jobs, Columns: salary, location, etc."

    FUNCTION run(user_query):
        # Step 1: Generate SQL
        schema = self.get_schema_info()
        prompt = f"Given schema {schema}, write a SQL query for: {user_query}"
        sql_query = self.llm.predict(prompt)
        
        # Step 2: Safety Check
        IF "DROP" IN sql_query OR "DELETE" IN sql_query:
            RETURN "Maaf, saya hanya diperbolehkan membaca data."

        # Step 3: Execute SQL
        TRY:
            raw_result = self.db.execute(sql_query)
        CATCH Error:
            RETURN "Gagal mengambil data, query SQL mungkin salah."

        # Step 4: Humanize Result
        final_answer = self.llm.predict(f"User asked: {user_query}. Data found: {raw_result}. Summarize this for user.")
        
        RETURN final_answer
"""
