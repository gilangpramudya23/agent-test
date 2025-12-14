"""
SQL Agent untuk query data terstruktur dari database SQLite
Menggunakan LangChain SQL Agent dengan safety features
"""

import os
import logging
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit  # ✅ FIXED

logger = logging.getLogger(__name__)

class SQLAgent:
    """Agent untuk query database SQL dengan natural language"""
    
    def __init__(
        self, 
        db_uri: Optional[str] = None,
        llm_model: str = "gpt-3.5-turbo",
        max_iterations: int = 5,
        verbose: bool = False
    ):
        """
        Inisialisasi SQL Agent dengan database connection
        
        Args:
            db_uri: URI database SQLite (default: auto-detect)
            llm_model: Model LLM untuk SQL generation
            max_iterations: Maksimal iterasi reasoning
            verbose: Tampilkan log detail
        """
        # Setup database URI (auto-detection untuk project structure)
        if db_uri is None:
            db_uri = self._auto_detect_db_path()
        
        # Initialize database connection
        try:
            self.db = SQLDatabase.from_uri(
                db_uri,
                include_tables=['jobs', 'salaries', 'companies'],  # Hanya table yang diperlukan
                sample_rows_in_table_info=3,  # Ambil sample data untuk context
            )
            logger.info(f"Connected to database: {db_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
        
        # Setup LLM dengan temperature 0 untuk konsistensi SQL
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Setup toolkit dan agent executor
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        
        # Create SQL agent dengan safety features
        self.agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=toolkit,
            verbose=verbose,
            max_iterations=max_iterations,
            handle_parsing_errors=True,  # Self-correction
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "return_intermediate_steps": False
            },
            prefix="""
            Anda adalah assistant yang ahli dalam database lowongan pekerjaan Indonesia.
            
            ATURAN PENTING:
            1. HANYA lakukan QUERY SELECT (baca data)
            2. JANGAN pernah eksekusi DROP, DELETE, INSERT, UPDATE
            3. Jika query error, analisis error dan perbaiki
            4. Berikan jawaban dalam bahasa Indonesia yang natural
            5. Format angka dengan pemisah ribuan (contoh: 10.000.000)
            
            Database berisi tabel:
            - jobs: lowongan pekerjaan dengan kolom: id, title, company, location, salary_min, salary_max, work_type, posted_date
            - salaries: data gaji berdasarkan lokasi dan role
            - companies: informasi perusahaan
            
            Contoh query yang valid:
            User: "Rata-rata gaji di Jakarta"
            SQL: SELECT AVG((salary_min + salary_max) / 2) FROM jobs WHERE location LIKE '%Jakarta%'
            
            User: "Top 5 perusahaan dengan gaji tertinggi"
            SQL: SELECT company, AVG((salary_min + salary_max) / 2) as avg_salary FROM jobs GROUP BY company ORDER BY avg_salary DESC LIMIT 5
            """,
        )
    
    def _auto_detect_db_path(self) -> str:
        """
        Auto-detect database path berdasarkan project structure
        
        Returns:
            SQLite database URI
        """
        # Coba beberapa lokasi umum
        possible_paths = [
            # Dari root project
            os.path.join(os.getcwd(), 'data', 'processed', 'jobs.db'),
            # Dari folder agents
            os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'jobs.db'),
            # Absolute path fallback
            '/app/data/processed/jobs.db'  # Untuk Docker deployment
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # Convert ke URI format
                absolute_path = os.path.abspath(path)
                return f"sqlite:///{absolute_path}"
        
        # Jika tidak ditemukan, buat database minimal
        logger.warning("Database not found, creating minimal database...")
        return self._create_minimal_database()
    
    def _create_minimal_database(self) -> str:
        """Create minimal database jika tidak ada"""
        import sqlite3
        from pathlib import Path
        
        # Buat direktori jika belum ada
        db_dir = os.path.join(os.getcwd(), 'data', 'processed')
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        
        db_path = os.path.join(db_dir, 'jobs.db')
        conn = sqlite3.connect(db_path)
        
        # Create minimal tables
        conn.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY,
            title TEXT,
            company TEXT,
            location TEXT,
            salary_min INTEGER,
            salary_max INTEGER,
            work_type TEXT,
            posted_date DATE,
            description TEXT
        )
        ''')
        
        # Insert sample data
        sample_jobs = [
            ("Software Engineer", "Tech Corp", "Jakarta", 15000000, 25000000, "Full-time", "2024-01-15"),
            ("Data Analyst", "Data Inc", "Bandung", 12000000, 18000000, "Hybrid", "2024-01-10"),
            ("Product Manager", "Startup XYZ", "Remote", 20000000, 35000000, "Remote", "2024-01-05")
        ]
        
        conn.executemany(
            "INSERT INTO jobs (title, company, location, salary_min, salary_max, work_type, posted_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            sample_jobs
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Created minimal database at: {db_path}")
        return f"sqlite:///{os.path.abspath(db_path)}"
    
    def run(self, query: str) -> str:
        """
        Eksekusi query natural language ke SQL
        
        Args:
            query: Pertanyaan dalam natural language
            
        Returns:
            Jawaban berdasarkan data database
        """
        try:
            # Safety check: cegah query berbahaya
            dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'ALTER', 'TRUNCATE']
            if any(keyword in query.upper() for keyword in dangerous_keywords):
                return "Maaf, saya hanya diperbolehkan membaca data, tidak bisa mengubah atau menghapus data."
            
            # Eksekusi melalui agent
            response = self.agent_executor.invoke({"input": query})
            
            # Extract answer dari response
            if isinstance(response, dict) and "output" in response:
                answer = response["output"]
            else:
                answer = str(response)
            
            # Clean up dan format answer
            answer = self._format_answer(answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"SQL Agent error: {str(e)}")
            return (
                f"Maaf, terjadi kesalahan saat mengakses database: {str(e)[:100]}... "
                "Silakan coba dengan pertanyaan yang lebih spesifik."
            )
    
    def _format_answer(self, answer: str) -> str:
        """Format jawaban agar lebih readable"""
        # Hapus prefix yang tidak perlu
        prefixes = ["Final Answer: ", "Answer: ", "Jawaban: "]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):]
        
        # Tambahkan disclaimer
        answer += "\n\n📊 *Data berdasarkan database lowongan pekerjaan Indonesia*"
        
        return answer

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
