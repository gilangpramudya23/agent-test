"""
app.py - Streamlit UI untuk AI Career Assistant Indonesia
Antarmuka pengguna untuk sistem multi-agent career counseling
"""

# ==================== STEP 1: IMPORTS & SETUP DASAR ====================
import streamlit as st
import os
import sys
from pathlib import Path
import tempfile
import time
from datetime import datetime

# Tambahkan path agents ke sys.path
sys.path.append(str(Path(__file__).parent / "agents"))

# Import agents dari package kita
from agents import RAGAgent, SQLAgent, AdvisorAgent, Orchestrator

# Import untuk environment variables
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Career Assistant Indonesia",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== STEP 2: INITIALIZATION DENGAN CACHING ====================
@st.cache_resource(show_spinner=False)
def initialize_agents():
    """
    Initialize semua agent dengan caching.
    Hanya dijalankan sekali saat pertama kali aplikasi dimuat.
    
    Returns:
        Tuple: (rag_agent, sql_agent, advisor_agent, orchestrator)
    """
    try:
        st.sidebar.info("🔄 Menginisialisasi AI Agents...")
        
        # 1. Initialize RAG Agent
        with st.spinner("Menghubungkan ke database lowongan..."):
            rag_agent = RAGAgent(
                collection_name="indonesian_jobs",
                embedding_model="text-embedding-3-small",
                llm_model="gpt-4o-mini"
            )
        
        # 2. Initialize SQL Agent  
        with st.spinner("Menghubungkan ke database analitik..."):
            sql_agent = SQLAgent(
                llm_model="gpt-3.5-turbo",
                verbose=False
            )
        
        # 3. Initialize Advisor Agent (butuh RAG Agent)
        with st.spinner("Menyiapkan konsultan karir..."):
            advisor_agent = AdvisorAgent(
                rag_agent=rag_agent,
                llm_model="gpt-4o-mini",
                temperature=0.7
            )
        
        # 4. Initialize Orchestrator (pusat kendali)
        with st.spinner("Menyiapkan sistem routing..."):
            orchestrator = Orchestrator(
                rag_agent=rag_agent,
                sql_agent=sql_agent,
                advisor_agent=advisor_agent,
                llm_model="gpt-4o-mini",
                temperature=0
            )
        
        st.sidebar.success("✅ Sistem siap!")
        return rag_agent, sql_agent, advisor_agent, orchestrator
        
    except Exception as e:
        st.sidebar.error(f"❌ Gagal menginisialisasi: {str(e)[:100]}")
        logger.error(f"Initialization error: {str(e)}")
        return None, None, None, None

# ==================== STEP 3: SIDEBAR & NAVIGATION ====================
def setup_sidebar():
    """
    Setup sidebar dengan navigation dan informasi
    """
    with st.sidebar:
        st.title("🎯 AI Career Assistant")
        st.markdown("---")
        
        # Mode Selection
        st.subheader("📋 Mode Konsultasi")
        mode = st.radio(
            "Pilih layanan:",
            ["💬 Tanya Lowongan", "📊 Analisis Data", "📄 Analisis CV", "ℹ️ Tentang"]
        )
        
        st.markdown("---")
        
        # Statistics/Info Panel
        st.subheader("📈 Statistik")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Lowongan Tersedia", "1,200+", "↗️ 12%")
        with col2:
            st.metric("Perusahaan", "350+", "↗️ 8%")
        
        st.markdown("---")
        
        # Quick Tips
        st.subheader("💡 Tips Cepat")
        st.info("""
        • Gunakan kata kunci spesifik: "Backend Developer Jakarta"
        • Untuk analisis data: "Gaji rata-rata Data Scientist"
        • Upload CV untuk rekomendasi personalisasi
        """)
        
        st.markdown("---")
        
        # System Status
        st.subheader("🔧 Status Sistem")
        if all(agent is not None for agent in [rag_agent, sql_agent, advisor_agent, orchestrator]):
            st.success("Semua sistem berjalan normal")
        else:
            st.error("Beberapa sistem bermasalah")
        
        # Footer
        st.markdown("---")
        st.caption("v1.0 • © 2024 AI Career Assistant")
        
        return mode

# ==================== STEP 8: UTILITY FUNCTIONS ====================
def export_chat_history():
    """
    Export chat history ke file teks
    """
    if not st.session_state.get("chat_history"):
        st.warning("Tidak ada chat history untuk diexport")
        return
    
    # Format chat history
    export_content = "AI Career Assistant - Chat History\n"
    export_content += f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    export_content += "=" * 50 + "\n\n"
    
    for role, message in st.session_state.chat_history:
        prefix = "👤 User: " if role == "user" else "🤖 AI: "
        export_content += f"{prefix}{message}\n\n"
    
    # Create download button
    st.download_button(
        label="📥 Download Chat History",
        data=export_content,
        file_name=f"career_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def show_example_queries():
    """
    Tampilkan contoh pertanyaan yang bisa diajukan
    """
    examples = [
        "Cari lowongan backend developer di Jakarta dengan gaji di atas 10 juta",
        "Apa persyaratan untuk menjadi data scientist?",
        "Perusahaan apa yang sedang mencari product manager?",
        "Berapa rata-rata gaji software engineer di Bali?",
        "Skill apa yang paling dicari untuk digital marketing?",
        "Tampilkan lowongan remote untuk graphic designer",
        "Bagaimana perkembangan karir untuk fresh graduate IT?",
        "Apa bedanya frontend dan backend developer?"
    ]
    
    st.info("💡 **Contoh Pertanyaan:**")
    for example in examples:
        if st.button(example, key=f"example_{example[:10]}", use_container_width=True):
            # Auto-fill chat input (tidak langsung di Streamlit, butuh workaround)
            st.session_state.chat_input = example
            st.rerun()

def check_environment():
    """
    Cek apakah environment variables sudah di-set
    """
    required_vars = ["OPENAI_API_KEY", "QDRANT_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        st.error(f"⚠️ Environment variables berikut belum di-set: {', '.join(missing_vars)}")
        st.info("""
        Tambahkan ke file `.env`:
        ```
        OPENAI_API_KEY=sk-...
        QDRANT_URL=https://your-cluster.cloud.qdrant.io
        QDRANT_API_KEY=your-api-key
        ```
        """)
        return False
    return True

# ==================== STEP 4: CHAT INTERFACE ====================
def render_chat_mode(orchestrator):
    """
    Render mode chat untuk tanya lowongan umum
    
    Args:
        orchestrator: Instance orchestrator untuk routing
    """
    st.header("💬 Tanya Lowongan & Karir")
    st.markdown("Tanyakan apapun tentang lowongan, perusahaan, atau karir di Indonesia")
    
    # Initialize session state untuk chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Tanya tentang lowongan atau karir..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_history.append(("user", prompt))
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Mencari informasi..."):
                try:
                    # Get response dari orchestrator
                    response = orchestrator.route_query(prompt)
                    
                    # Display response dengan streaming effect
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    # Simulate streaming
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    
                    message_placeholder.markdown(full_response)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.chat_history.append(("assistant", response))
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)[:200]}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            export_chat_history()
    with col3:
        if st.button("🎯 Contoh Pertanyaan", use_container_width=True):
            show_example_queries()

# ==================== STEP 5: CV ANALYSIS MODE ====================
def render_cv_mode(advisor_agent):
    """
    Render mode untuk upload dan analisis CV
    
    Args:
        advisor_agent: Instance advisor agent untuk analisis CV
    """
    st.header("📄 Analisis CV & Rekomendasi Karir")
    st.markdown("Upload CV Anda (PDF) untuk mendapatkan analisis dan rekomendasi lowongan yang cocok")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CV Anda (PDF)",
        type=["pdf"],
        help="Maksimal 10MB. File akan diproses secara aman dan tidak disimpan permanen."
    )
    
    if uploaded_file is not None:
        # Display file info
        file_details = {
            "Nama File": uploaded_file.name,
            "Tipe File": uploaded_file.type,
            "Ukuran": f"{uploaded_file.size / 1024:.2f} KB"
        }
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("📋 **Detail File:**")
            for key, value in file_details.items():
                st.write(f"{key}: {value}")
        
        with col2:
            # Preview PDF (basic)
            st.write("👁️ **Preview:**")
            st.info("File PDF terdeteksi. Klik 'Analisis CV' untuk memulai.")
        
        # Analysis button
        if st.button("🔍 Analisis CV", type="primary", use_container_width=True):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # Show progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Processing file
                status_text.text("📥 Memproses file CV...")
                progress_bar.progress(25)
                time.sleep(1)
                
                # Step 2: Extracting text
                status_text.text("🔤 Mengekstrak teks dari CV...")
                progress_bar.progress(50)
                time.sleep(1)
                
                # Step 3: Analyzing profile
                status_text.text("🧠 Menganalisis profil dan skill...")
                progress_bar.progress(75)
                
                # Actually analyze with advisor agent
                with st.spinner("Menganalisis CV dan mencari lowongan yang cocok..."):
                    analysis_result = advisor_agent.analyze_and_recommend(tmp_path)
                
                progress_bar.progress(100)
                status_text.text("✅ Analisis selesai!")
                time.sleep(0.5)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Hasil Analisis CV")
                
                # Display in expandable sections
                with st.expander("📄 **Rekomendasi Lengkap**", expanded=True):
                    st.markdown(analysis_result)
                
                # Additional suggestions
                st.subheader("🎯 Langkah Selanjutnya")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.info("""
                    **✍️ Perbaiki CV**
                    - Optimasi kata kunci
                    - Highlight pencapaian
                    - Format profesional
                    """)
                
                with col2:
                    st.info("""
                    **📚 Tingkatkan Skill**
                    - Identifikasi skill gap
                    - Rekomendasi kursus
                    - Project portfolio
                    """)
                
                with col3:
                    st.info("""
                    **🤝 Network**
                    - Hubungi recruiter
                    - LinkedIn optimization
                    - Networking events
                    """)
                
                # Clean up temp file
                os.unlink(tmp_path)
                
            except Exception as e:
                st.error(f"❌ Error dalam analisis: {str(e)[:200]}")
                logger.error(f"CV analysis error: {str(e)}")
    
    # CV Tips Section
    with st.expander("💡 Tips CV yang Baik"):
        st.markdown("""
        **Do's:**
        - Gunakan format PDF
        - Sertakan kata kunci skill
        - Tunjukkan pencapaian dengan angka
        - Max 2 halaman
        
        **Don'ts:**
        - Foto tidak profesional
        - Typo dan grammar error
        - Format tidak konsisten
        - Informasi tidak relevan
        """)

# ==================== STEP 6: DATA ANALYSIS MODE ====================
def render_data_mode(sql_agent):
    """
    Render mode untuk analisis data lowongan
    
    Args:
        sql_agent: Instance SQL agent untuk query data
    """
    st.header("📊 Analisis Data Pasar Kerja")
    st.markdown("Analisis statistik, trend, dan insights dari database lowongan Indonesia")
    
    # Quick analysis buttons
    st.subheader("🚀 Analisis Cepat")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_queries = {
        "💰 Gaji Rata-rata": "Berapa rata-rata gaji untuk software engineer di Indonesia?",
        "📍 Lokasi Populer": "Kota apa yang memiliki paling banyak lowongan IT?",
        "🏢 Perusahaan Top": "Perusahaan apa yang paling banyak membuka lowongan?",
        "📈 Trend Remote": "Berapa persentase lowongan yang remote vs hybrid vs onsite?"
    }
    
    query_result = None
    
    for col, (title, query) in zip([col1, col2, col3, col4], quick_queries.items()):
        with col:
            if st.button(title, use_container_width=True):
                query_result = query
    
    # Custom query input
    st.subheader("🔍 Query Kustom")
    custom_query = st.text_area(
        "Tanya apapun tentang data lowongan:",
        placeholder="Contoh: 'Tampilkan 5 skill yang paling banyak dicari untuk data scientist'",
        height=100
    )
    
    # Execute query
    if query_result or custom_query:
        query_to_execute = query_result if query_result else custom_query
        
        with st.spinner(f"Menganalisis: '{query_to_execute[:50]}...'"):
            try:
                result = sql_agent.run(query_to_execute)
                
                # Display result
                st.markdown("---")
                st.subheader("📋 Hasil Analisis")
                
                # Format result berdasarkan tipe
                if "SELECT" in query_to_execute.upper() or "rata" in query_to_execute.lower():
                    # Jika query numerik/statistik
                    st.info(result)
                    
                    # Visualisasi sederhana (jika ada angka)
                    if any(word in result for word in ["%", "persen", "rata-rata", "jumlah"]):
                        # Contoh visualisasi sederhana
                        st.caption("📈 Visualisasi Data:")
                        # Di sini bisa ditambahkan chart berdasarkan parsing result
                        
                else:
                    # General response
                    st.write(result)
                
                # Query explanation
                with st.expander("ℹ️ Penjelasan Query"):
                    st.markdown(f"""
                    **Query Anda:** {query_to_execute}
                    
                    **Jenis Analisis:** {'Statistik' if 'rata' in query_to_execute.lower() else 'Informasi'}
                    
                    **Sumber Data:** Database lowongan pekerjaan Indonesia
                    
                    **Catatan:** Data diperbarui secara berkala. Hasil merupakan estimasi berdasarkan data yang tersedia.
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error dalam analisis data: {str(e)[:200]}")
    
    # Sample datasets preview
    with st.expander("📁 Preview Data Tersedia"):
        st.markdown("""
        **Tabel jobs:**
        - title: Judul lowongan
        - company: Nama perusahaan
        - location: Lokasi kerja
        - salary_min, salary_max: Range gaji
        - work_type: Remote/Hybrid/Onsite
        - posted_date: Tanggal posting
        
        **Contoh Query SQL:**
        ```sql
        -- Rata-rata gaji per kota
        SELECT location, AVG((salary_min + salary_max) / 2) as avg_salary
        FROM jobs 
        GROUP BY location 
        ORDER BY avg_salary DESC
        LIMIT 10;
        ```
        """)

# ==================== STEP 7: ABOUT & HELP MODE ====================
def render_about_mode():
    """
    Render mode tentang/tentang aplikasi
    """
    st.header("ℹ️ Tentang AI Career Assistant")
    
    # App description
    st.markdown("""
    ## 🚀 Platform Konsultasi Karir Berbasis AI
    
    **AI Career Assistant** adalah sistem multi-agent yang membantu pencari kerja di Indonesia 
    dengan berbagai layanan berbasis kecerdasan artifisial.
    """)
    
    # Features grid
    st.subheader("✨ Fitur Unggulan")
    
    col1, col2, col3 = st.columns(3)
    
    features = [
        ("🔍 Semantic Search", "Cari lowongan dengan makna, bukan hanya kata kunci"),
        ("📊 Data Analytics", "Analisis trend pasar kerja dan gaji"),
        ("🤖 CV Analysis", "Analisis CV dan rekomendasi personalisasi"),
        ("💬 Chat Expert", "Konsultasi karir dengan AI consultant"),
        ("📈 Career Path", "Rekomendasi perkembangan karir"),
        ("🔒 Privacy First", "Data Anda aman dan tidak disimpan permanen")
    ]
    
    for i, (title, desc) in enumerate(features):
        col = [col1, col2, col3][i % 3]
        with col:
            st.info(f"**{title}**\n\n{desc}")
    
    # Technology stack
    st.subheader("🛠️ Teknologi yang Digunakan")
    
    tech_cols = st.columns(4)
    tech_stack = [
        ("OpenAI", "GPT-4o, Embeddings"),
        ("Qdrant", "Vector Database"),
        ("LangChain", "Agent Framework"),
        ("Streamlit", "UI Framework"),
        ("SQLite", "Database"),
        ("Docker", "Containerization"),
        ("Python", "Backend"),
        ("PyPDF", "PDF Processing")
    ]
    
    for i, (name, desc) in enumerate(tech_stack):
        with tech_cols[i % 4]:
            st.markdown(f"""
            <div style='padding: 10px; border-radius: 5px; background-color: #f0f2f6; margin: 5px;'>
                <b>{name}</b><br>
                <small>{desc}</small>
            </div>
            """, unsafe_allow_html=True)
    
    # FAQ
    with st.expander("❓ FAQ - Pertanyaan Umum"):
        faq_items = [
            ("Apa perbedaan dengan job portal biasa?", 
             "Kami menggunakan AI untuk memahami konteks dan makna, bukan hanya keyword matching."),
            ("Apakah data saya aman?", 
             "Ya, file CV diproses sementara dan tidak disimpan permanen. Chat history hanya tersimpan di sesi browser Anda."),
            ("Berapa akurasi rekomendasi?", 
             "Akurasi tergantung kualitas data dan query. Untuk hasil terbaik, gunakan kata kunci spesifik."),
            ("Bisakah untuk fresh graduate?", 
             "Tentu! Sistem kami memiliki mode khusus untuk fresh graduate dan career starter.")
        ]
        
        for question, answer in faq_items:
            st.markdown(f"**Q:** {question}")
            st.markdown(f"**A:** {answer}")
            st.markdown("---")

# ==================== STEP 9: MAIN APPLICATION LOGIC ====================
def main():
    """
    Main function untuk menjalankan Streamlit app
    """
    # Check environment first
    if not check_environment():
        return
    
    # Initialize agents dengan caching
    rag_agent, sql_agent, advisor_agent, orchestrator = initialize_agents()
    
    # Jika inisialisasi gagal, tampilkan error
    if orchestrator is None:
        st.error("""
        ❌ Gagal menginisialisasi sistem. Kemungkinan penyebab:
        1. API keys tidak valid
        2. Koneksi database gagal
        3. Network error
        
        Periksa:
        - File `.env` sudah benar?
        - Koneksi internet stabil?
        - API quota masih tersedia?
        """)
        return
    
    # Setup sidebar dan dapatkan mode yang dipilih
    mode = setup_sidebar()
    
    # Main content area berdasarkan mode
    if mode == "💬 Tanya Lowongan":
        render_chat_mode(orchestrator)
    
    elif mode == "📊 Analisis Data":
        render_data_mode(sql_agent)
    
    elif mode == "📄 Analisis CV":
        render_cv_mode(advisor_agent)
    
    elif mode == "ℹ️ Tentang":
        render_about_mode()

# ==================== RUN THE APP ====================
if __name__ == "__main__":
    # Custom CSS untuk styling tambahan
    st.markdown("""
    <style>
    /* Custom styling */
    .stButton button {
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Run main app
    main()