# src/app/streamlit_app.py
"""
MINIMAL UI LAYER - All business logic moved to services/
"""
from pathlib import Path
import sys
import streamlit as st

# --- Make sure the project root is on sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import SERVICES (business logic layer) instead of models directly
from src.services.search_service import SearchService
from src.services.topic_service import TopicService
from src.services.summary_service import SummaryService
from src.services.qa_service import QAService


# -----------------------
# Service Initialization (singleton pattern)
# -----------------------
@st.cache_resource
def init_services():
    """Initialize all services once per session"""
    return {
        "search": SearchService(),
        "topics": TopicService(),
        "summary": SummaryService(),
        "qa": QAService()
    }


# -----------------------
# Session State
# -----------------------
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "search_results": None,      # List of Paper objects
        "selected_paper_id": None,   # ID of selected paper
        "current_query": "",         # Last search query
        "search_mode": "keyword",    # Current search mode
        "topic_data": None,          # Topic analysis results
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------
# UI Components (pure display logic)
# -----------------------
def render_search_controls():
    """Render search input and controls"""

    # Search mode with bigger label
    st.markdown("""
    <div style="margin-bottom: 0.3rem; font-weight: 600; font-size: 1.1rem; color: #e5679a;">
    Search mode
    </div>
    """, unsafe_allow_html=True)
    
    search_mode = st.radio(
        "Search mode",
        ["Keyword (fast)", "Semantic (embeddings)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="search_mode_radio"
    )
    
    # Semantic search explanation - ALWAYS VISIBLE
    st.markdown("""
    <div style='background-color: #2a1a3a; padding: 8px 12px; border-radius: 6px; border-left: 4px solid #e5679a; margin: 8px 0 12px 0; font-size: 0.9em; color: #f0e6f6;'>
    <strong style="color: #e5679a;">Semantic search</strong> finds papers based on meaning using embeddings. Better for complex topics.
    </div>
    """, unsafe_allow_html=True)
    
    query = st.text_input(
        "Research topic",
        placeholder="e.g. machine learning, diffusion models, federated learning…",
        label_visibility="collapsed"
    )
    
    arxiv_category = st.text_input(
        "Optional arXiv category (e.g., cs.LG, cs.CL, stat.ML). Leave empty for broad search.",
        value="",
    )
    
    # Compact button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.empty()  # Spacer
    with col2:
        submitted = st.button("Search arXiv", type="primary")
    
    # Placeholder message goes HERE (below search mode)
    if not query:
        st.info("🔍 Enter a research topic above to search for papers.")
    
    return {
        "query": query,
        "submitted": submitted,
        "mode": "keyword" if search_mode.startswith("Keyword") else "semantic",
        "category": arxiv_category if arxiv_category else None
    }


def render_paper_card(paper, index, services, max_len, min_len):
    """Render a single paper result card"""
    title = paper.title
    if hasattr(paper, 'similarity') and paper.similarity is not None:
        title = f"{title}  (similarity: {paper.similarity:.3f})"
    
    with st.expander(title):
        # Metadata
        meta = f"**Authors:** {paper.authors}"
        if paper.published:
            meta += f"  |  **Published:** {paper.published}"
        st.markdown(meta)
        
        # Links
        links = []
        if hasattr(paper, 'entry_id') and paper.entry_id:
            links.append(f"[Open on arXiv]({paper.entry_id})")
        if paper.pdf_url:
            links.append(f"[Read paper (PDF)]({paper.pdf_url})")
        if links:
            st.markdown("  |  ".join(links))
        
        # Abstract
        st.markdown("**Abstract (arXiv)**")
        st.write(paper.abstract)
        
        # Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select for Topic Insights / Q&A", key=f"sel_{paper.id}"):
                st.session_state.selected_paper_id = paper.id
                st.success("Selected. Go to Topic Insights or Q&A tab.")
        with col2:
            if st.button("Summarise now", key=f"sum_{paper.id}"):
                with st.spinner("Generating summary…"):
                    summary = services["summary"].summarize_paper(
                        paper, 
                        max_length=max_len, 
                        min_length=min_len
                    )
                st.markdown("**Summary (generated)**")
                st.write(summary)


def render_topics_tab(services):
    """Render topics analysis tab - PURE UI LOGIC"""
    st.header("📊 Topic Insights")
    
    if not st.session_state.search_results:
        st.info("Run a search in Summaries to generate topics.")
        return
    
    papers = st.session_state.search_results
    
    # Check minimum papers requirement (UI logic only)
    if len(papers) < 5:
        st.info(f"Need at least 5 papers for topic modeling. Found {len(papers)}.")
        
        # Show simple paper list (UI only)
        if len(papers) > 0:
            st.subheader("Papers found:")
            for i, paper in enumerate(papers):
                st.write(f"{i+1}. **{paper.title}**")
        return
    
    # Generate topics if not already done
    if not st.session_state.topic_data:
        with st.spinner("Analyzing topics…"):
            # Call business logic service
            topic_data = services["topics"].analyze_live_results(papers)
            
            if topic_data.get("success"):
                st.session_state.topic_data = topic_data
                st.success(f"✅ Generated {topic_data.get('num_topics', 0)} topics from {len(papers)} papers!")
            else:
                error_msg = topic_data.get("error", "Could not generate topics")
                st.warning(error_msg)
                # Provide helpful suggestions
                if "generic" in error_msg.lower() or "meaningful" in error_msg.lower():
                    st.info("💡 **Tip**: Try a more specific search query or search for more papers.")
                st.session_state.topic_data = {"error": error_msg}
                return
    
    topic_data = st.session_state.topic_data
    
    # Check for errors (UI logic)
    if "error" in topic_data:
        st.warning(topic_data["error"])
        return
    
    if not topic_data.get("success"):
        st.info("Could not generate topics. Try a larger search.")
        return
    
    # Display topics from service (UI only)
    if "topics_df" in topic_data and not topic_data["topics_df"].empty:
        topics_df = topic_data["topics_df"]
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Topics", len(topics_df))
        with col2:
            st.metric("Total Papers", len(papers))
        with col3:
            # Show topic names instead of average
            topic_names = topics_df['topic_name'].head(3).tolist()
            st.markdown("**📑 Topics Found:**")
            for i, name in enumerate(topic_names, 1):
                st.caption(f"{i}. {name}")
            if len(topics_df) > 3:
                st.caption(f"... and {len(topics_df) - 3} more")
        
        st.markdown("---")
        st.subheader("🔬 Explore Topics")
        
        # Create a dropdown with better topic names
        topic_options = []
        for idx, row in topics_df.iterrows():
            topic_id = row['topic_id']
            topic_name = row.get('topic_name', f"Topic {topic_id}")
            doc_count = row['doc_count']
            topic_options.append((topic_id, f"{topic_name} ({doc_count} papers)"))
        
        # Let user select a topic to explore
        selected_option = st.selectbox(
            "Choose a topic to explore in detail:",
            options=[opt[1] for opt in topic_options],
            index=0
        )
        
        # Find the selected topic ID
        selected_topic_id = None
        for topic_id, option_text in topic_options:
            if option_text == selected_option:
                selected_topic_id = topic_id
                break
        
        if selected_topic_id is not None:
            # Get papers in the selected topic
            selected_papers = services["topics"].get_papers_in_topic(
                selected_topic_id, 
                papers, 
                topic_data
            )
            
            if selected_papers:
                st.markdown(f"### 📄 Papers in '{selected_option}'")
                
                # Display selected topic's keywords
                selected_topic_row = topics_df[topics_df['topic_id'] == selected_topic_id].iloc[0]
                selected_keywords = selected_topic_row['keywords'].split(', ')
                
                # Show keywords properly without duplication
                keyword_badges = ""
                for i, kw in enumerate(selected_keywords[:5]):
                    colors = ['#e5679a', '#da3f59', '#73378a', '#5c038c', '#e5679a']
                    color = colors[min(i, len(colors)-1)]
                    keyword_badges += f'<span style="background-color: {color}; color: white; padding: 6px 12px; margin: 3px; border-radius: 20px; font-size: 0.88em; display: inline-block; font-weight: 500; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">{kw}</span> '
                
                st.markdown(f"**Key terms:** <br>{keyword_badges}", unsafe_allow_html=True)
                
                # Show all papers in this topic
                for paper in selected_papers:
                    with st.container(border=True):
                        st.markdown(f"#### {paper.title}")
                        
                        # Paper details in columns
                        col_info, col_action = st.columns([4, 1])
                        
                        with col_info:
                            if hasattr(paper, 'authors'):
                                st.markdown(f"**Authors:** {paper.authors}")
                            if hasattr(paper, 'published'):
                                st.caption(f"📅 Published: {paper.published}")
                            
                            # Abstract preview
                            with st.expander("Abstract preview"):
                                st.write(paper.abstract[:300] + "..." if len(paper.abstract) > 300 else paper.abstract)
                        
                        with col_action:
                            if st.button("🔗 Select", key=f"explore_{paper.id}", 
                                        help="Select this paper for Q&A", type="primary"):
                                st.session_state.selected_paper_id = paper.id
                                st.success("✅ Paper selected! Go to Q&A tab.")
                                st.rerun()
                
                # Download option for this topic's papers
                st.download_button(
                    label="📥 Download Topic Papers",
                    data="\n\n".join([f"Title: {p.title}\nAuthors: {p.authors}\nAbstract: {p.abstract[:200]}..." 
                                     for p in selected_papers]),
                    file_name=f"topic_{selected_topic_id}_papers.txt",
                    mime="text/plain"
                )
            else:
                st.info("No papers found in the selected topic.")
        
    else:
        st.info("No topics generated. Try a different search.")


def render_qa_tab(services):
    """Render Q&A tab"""
    st.header("❓ Q&A from Paper PDF")
    
    if not st.session_state.search_results:
        st.info("Run a search first in Summaries.")
        return
    
    if not st.session_state.selected_paper_id:
        st.info("Select a paper in Summaries or Topics tab to ask questions about it.")
        return
    
    # Find selected paper
    selected_paper = None
    for paper in st.session_state.search_results:
        if paper.id == st.session_state.selected_paper_id:
            selected_paper = paper
            break
    
    if not selected_paper:
        st.warning("Selected paper not found in current results. Please search again and re-select.")
        return
    
    # Display paper info
    st.subheader(selected_paper.title)
    if selected_paper.authors:
        st.caption(selected_paper.authors)
    
    if hasattr(selected_paper, 'entry_id') and selected_paper.entry_id:
        st.markdown(f"[Open on arXiv]({selected_paper.entry_id})")
    if selected_paper.pdf_url:
        st.markdown(f"[Read paper (PDF)]({selected_paper.pdf_url})")
    
    if not selected_paper.pdf_url or not str(selected_paper.pdf_url).strip():
        st.error("No PDF URL available for this paper.")
        return
    
    # Q&A interface
    st.markdown("### Ask a question about this paper")
    question = st.text_area("Enter your question:", placeholder="e.g., What methodology did the authors use? What were the main findings?", height=100)
    
    col1, col2 = st.columns([1, 6])
    with col1:
        ask_button = st.button("🤖 Get Answer", type="primary", use_container_width=True)
    
    if ask_button:
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching paper and generating answer…"):
                answer = services["qa"].answer_question(selected_paper, question)
            
            if answer.get("error"):
                st.warning(answer["error"])
            else:
                st.markdown("### Answer")
                st.write(answer.get("answer", ""))
                
                # Show metrics
                cols = st.columns(3)
                if "score" in answer:
                    with cols[0]:
                        st.metric("Confidence", f"{answer['score']:.2f}")
                if "similarity" in answer:
                    with cols[1]:
                        st.metric("Relevance", f"{answer['similarity']:.2f}")
                if "page" in answer:
                    with cols[2]:
                        st.metric("Page", answer["page"])
                
                # Show context if available
                if answer.get("context_snippet"):
                    with st.expander("📄 Source from paper"):
                        st.write(answer["context_snippet"])
                
                # Debug info
                if answer.get("chunks_used"):
                    with st.expander("🔍 Technical details"):
                        st.json(answer["chunks_used"])


# -----------------------
# Main App
# -----------------------
def main():
    """Main application entry point"""
    # Initialize
    init_session_state()
    services = init_services()
    
    # ============================================
    # PAGE CONFIGURATION
    # ============================================
    st.set_page_config(
        page_title="PAPERMINER - Research Paper Explorer",
        layout="wide",
        page_icon="🔍",
        initial_sidebar_state="expanded"
    )

    # ============================================
    # CUSTOM CSS - VIBRANT NIGHT MODE PALETTE
    # ============================================
    st.markdown("""
    <style>
    /* FORCE OVERRIDE - Streamlit's primary color */
    :root {
        --primary-color: #e5679a !important;
        --background-color: #0f0a1a !important;
        --secondary-background-color: #1a1225 !important;
    }
    
    /* Dark theme background */
    .stApp {
        background-color: #0f0a1a !important;
        margin-top: -60px !important;
        padding-top: 0 !important;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #0f0a1a !important;
        padding-top: 2rem !important;
    }
    
    /* Main header container */
    .paperminer-main {
        text-align: center;
        margin: -35px 0 0.5rem 0 !important;
        padding: 20px;
        background: linear-gradient(135deg, #1a1225 0%, #0f0a1a 100%);
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(229, 103, 154, 0.2);
        position: relative;
    }
    
    /* Main title - Hot Pink gradient */
    .paperminer-title {
        font-size: 3.8rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e5679a 0%, #da3f59 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 2px;
        margin-bottom: 0 !important;
        padding-top: 5px;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(229, 103, 154, 0.3);
    }
    
    /* Divider - Vibrant gradient */
    .divider {
        height: 4px;
        background: linear-gradient(90deg, #5c038c, #73378a, #da3f59, #e5679a) !important;
        width: 320px;
        margin: 0.3rem auto 0.4rem auto !important;
        border-radius: 4px;
        box-shadow: 0 2px 8px rgba(229, 103, 154, 0.4);
    }
    
    /* Subtitle - Hot Pink */
    .paperminer-subtitle {
        font-size: 1.3rem;
        color: #e5679a !important;
        font-weight: 500;
        margin-top: 0 !important;
        opacity: 0.95;
        text-shadow: 0 0 10px rgba(229, 103, 154, 0.3);
    }
    
    /* TABS - Vibrant gradient effect */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 14px 28px;
        margin: 0 4px;
        border-radius: 10px 10px 0 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }
    
    /* Tab 1 - Violet */
    .stTabs [data-baseweb="tab"]:first-child:not([aria-selected="true"]) {
        background: linear-gradient(135deg, #5c038c, #73378a) !important;
        color: white !important;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:first-child:not([aria-selected="true"]):hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(92, 3, 140, 0.5);
    }
    
    /* Tab 2 - Indigo to Red gradient */
    .stTabs [data-baseweb="tab"]:nth-child(2):not([aria-selected="true"]) {
        background: linear-gradient(135deg, #73378a, #da3f59) !important;
        color: white !important;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(2):not([aria-selected="true"]):hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(115, 55, 138, 0.5);
    }
    
    /* Tab 3 - Red to Pink gradient */
    .stTabs [data-baseweb="tab"]:nth-child(3):not([aria-selected="true"]) {
        background: linear-gradient(135deg, #da3f59, #e5679a) !important;
        color: white !important;
        border-bottom: none;
    }
    
    .stTabs [data-baseweb="tab"]:nth-child(3):not([aria-selected="true"]):hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(218, 63, 89, 0.5);
    }
    
    /* Selected tab - Hot Pink with glow */
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #e5679a, #da3f59) !important;
        color: white !important;
        font-weight: 700;
        border: 2px solid #e5679a;
        border-bottom: none;
        box-shadow: 0 0 20px rgba(229, 103, 154, 0.6), 0 4px 12px rgba(0,0,0,0.3);
        transform: translateY(-3px);
    }
    
    /* Tab content headers - Hot Pink with glow */
    h1, h2, h3, h4 {
        color: #e5679a !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
        text-shadow: 0 0 10px rgba(229, 103, 154, 0.3);
    }
    
    /* ========== RADIO BUTTON - Hot Pink ========== */
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"] > div,
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"] > div > div,
    div[data-baseweb="radio"] > div,
    div[data-baseweb="radio"] > div > div {
        border-color: #73378a !important;
    }
    
    /* Checked radio - Hot Pink with glow */
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"][aria-checked="true"] > div > div,
    div[data-baseweb="radio"] input:checked ~ div,
    div[data-baseweb="radio"] input:checked ~ div > div,
    div[data-baseweb="radio"] input:checked + div,
    div[data-baseweb="radio"] input:checked + div > div,
    div[data-baseweb="radio"] input:checked + div > div > div {
        background-color: #e5679a !important;
        border-color: #e5679a !important;
        box-shadow: 0 0 10px rgba(229, 103, 154, 0.6);
    }
    
    /* Radio labels */
    .stRadio label,
    [data-testid="stRadio"] label {
        color: #e5679a !important;
        font-weight: 500;
    }
    
    /* ========== BUTTONS - Vibrant gradients ========== */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #e5679a, #da3f59) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(229, 103, 154, 0.4);
    }
    
    button[kind="primary"]:hover,
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(229, 103, 154, 0.6);
    }
    
    /* Secondary buttons - Violet gradient */
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {
        background: linear-gradient(135deg, #5c038c, #73378a) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(92, 3, 140, 0.4);
    }
    
    button[kind="secondary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(92, 3, 140, 0.6);
    }
    
    /* Sidebar - Dark theme */
    [data-testid="stSidebar"] {
        background-color: #1a1225 !important;
    }
    
    /* Sidebar headers - Hot Pink */
    .sidebar .stMarkdown h3,
    [data-testid="stSidebar"] h3 {
        color: #e5679a !important;
        border-bottom: 2px solid #e5679a;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(229, 103, 154, 0.3);
    }
    
    .sidebar .stMarkdown h4,
    [data-testid="stSidebar"] h4 {
        color: #e5679a !important;
        font-weight: 600;
    }
    
    /* Slider labels - Hot Pink */
    .stSlider label {
        color: #e5679a !important;
        font-weight: 600 !important;
    }
    
    /* Slider track - Dark with violet tint */
    .stSlider > div > div > div[role="slider"],
    .stSlider [role="slider"] {
        background-color: #2a1a3a !important;
    }
    
    /* Slider handle - Hot Pink with glow */
    .stSlider > div > div > div > div,
    .stSlider [data-baseweb="slider"] [role="slider"] > div {
        background-color: #e5679a !important;
        border-color: #e5679a !important;
        box-shadow: 0 0 10px rgba(229, 103, 154, 0.6);
    }
    
    /* Checkbox - Hot Pink */
    .stCheckbox [data-baseweb="checkbox"] div:first-child,
    [data-testid="stCheckbox"] [role="checkbox"] {
        border-color: #e5679a !important;
    }
    
    .stCheckbox input:checked + div > div:first-child,
    [data-testid="stCheckbox"] input:checked ~ div {
        background-color: #e5679a !important;
        border-color: #e5679a !important;
        box-shadow: 0 0 8px rgba(229, 103, 154, 0.5);
    }
    
    /* Expander headers */
    .sidebar .streamlit-expanderHeader,
    .streamlit-expanderHeader {
        color: #e5679a !important;
        font-weight: 700 !important;
        background-color: #2a1a3a !important;
        border-radius: 6px;
    }
    
    /* Text colors for dark mode */
    * {
        color: #f0e6f6 !important;
    }
    
    /* Input fields - Dark theme */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background-color: #2a1a3a !important;
        color: #f0e6f6 !important;
        border: 1px solid #73378a !important;
        border-radius: 6px;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #e5679a !important;
        box-shadow: 0 0 10px rgba(229, 103, 154, 0.3);
    }
    
    /* Containers and cards */
    .element-container,
    [data-testid="stExpander"],
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #e5679a !important;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #d4b3e0 !important;
    }
    
    /* Info, warning, success boxes */
    .stAlert {
        background-color: #2a1a3a !important;
        border-left-color: #e5679a !important;
        color: #f0e6f6 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background-color: #1a1225 !important;
        border: 1px solid #73378a;
        border-radius: 0 0 8px 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============================================
    # PAGE HEADER
    # ============================================
    st.markdown("""
    <div class="paperminer-main">
        <h1 class="paperminer-title">🔍 PAPERMINER</h1>
        <div class="divider"></div>
        <p class="paperminer-subtitle">Scientific Literature Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # SIDEBAR
    # ============================================
    st.sidebar.markdown("### ⚙️ Settings")
    
    st.sidebar.markdown("#### Search Settings")
    top_k = st.sidebar.slider("Number of Papers to display", 5, 50, 10, step=5)
    max_len = st.sidebar.slider("Max summary length", 64, 256, 128, step=16)
    min_len = st.sidebar.slider("Min summary length", 16, 64, 32, step=8)
    
    with st.sidebar.expander("**💡 How to get better topics**", expanded=False):
        st.markdown("""
        - **Search for 10+ papers** for clearer topics
        - Use **specific queries** (e.g., "federated learning privacy" vs "machine learning")
        - Try different **arXiv categories** to focus your search
        - Generic words (data, model, using) are automatically filtered out
        """)
    
    with st.sidebar.expander("**ℹ️ About PAPERMINER**", expanded=False):
        st.markdown("""
        **PAPERMINER** helps you:
        
        1. **Search** arXiv papers by topic
        2. **Discover** research trends through topic analysis
        3. **Summarize** papers automatically
        4. **Ask questions** about full paper PDFs
        
        Built for researchers, students, and curious minds.
        """)

    # ============================================
    # MAIN TABS
    # ============================================
    tab_summaries, tab_topics, tab_qa = st.tabs(["📄 Papers & Summaries", "📊 Topic Insights", "❓ Paper Q&A"])
    
    # ============================================
    # SUMMARIES TAB - COMPACT
    # ============================================
    with tab_summaries:
        # Compact header
        st.markdown("""
        <div style="margin-bottom: 0.5rem;">
            <h2 style="margin: 0; padding: 0;">📄 Search & Paper Summaries</h2>
            <p style="margin: 0.1rem 0 0 0; color: #8b6a5a; font-size: 0.9rem;">
            Find arXiv papers and generate AI-powered summaries.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        search_inputs = render_search_controls()
        
        if search_inputs["submitted"] and search_inputs["query"]:
            with st.spinner("Searching arXiv…"):
                results = services["search"].search_arxiv(
                    topic=search_inputs["query"].strip(),
                    mode=search_inputs["mode"],
                    top_k=top_k,
                    category=search_inputs["category"]
                )
                
                if results:
                    st.session_state.search_results = results
                    st.session_state.current_query = search_inputs["query"]
                    st.session_state.search_mode = search_inputs["mode"]
                    st.session_state.selected_paper_id = None
                    st.session_state.topic_data = None
                    
                    st.success(f"✅ Found {len(results)} papers")
                    if len(results) >= 5:
                        st.info(f"✓ Enough papers for topic analysis! Go to **Topic Insights** tab.")
                    else:
                        st.warning(f"⚠ Need at least 5 papers for topic analysis. Found {len(results)}.")
                else:
                    st.info("No papers found for this query.")
        
        if st.session_state.search_results:
            st.markdown(f"### 📚 Showing {len(st.session_state.search_results)} Papers")
            for idx, paper in enumerate(st.session_state.search_results):
                render_paper_card(paper, idx, services, max_len, min_len)
    
    # ============================================
    # TOPICS TAB
    # ============================================
    with tab_topics:
        render_topics_tab(services)
    
    # ============================================
    # Q&A TAB
    # ============================================
    with tab_qa:
        render_qa_tab(services)
    
    # ============================================
    # FOOTER
    # ============================================
    st.markdown('<div style="height: 40px;"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()