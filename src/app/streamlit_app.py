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
    st.markdown("### 🔍 Search Papers")
    
    search_mode = st.radio(
        "Search mode",
        ["Keyword (fast)", "Semantic (embeddings)"],
        index=0,
        horizontal=True
    )
    
    query = st.text_input(
        "Research topic",
        placeholder="e.g. machine learning, diffusion models, federated learning…",
        label_visibility="collapsed"
    )
    
    arxiv_category = st.text_input(
        "Optional arXiv category (e.g., cs.LG, cs.CL, stat.ML). Leave empty for broad search.",
        value="",
    )
    
    submitted = st.button("Search arXiv", type="primary", use_container_width=True)
    
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
        
        # Clearer topic display
        st.subheader(f"📊 Found {len(topics_df)} Topics")
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Topics", len(topics_df))
        with col2:
            st.metric("Total Papers", len(papers))
        with col3:
            avg_papers = topics_df['doc_count'].mean()
            st.metric("Avg Papers/Topic", f"{avg_papers:.1f}")
        
        # Display each topic in a clear, visual way
        for idx, row in topics_df.iterrows():
            topic_id = row['topic_id']
            topic_name = row.get('topic_name', f"Topic {topic_id}")
            doc_count = row['doc_count']
            keywords = row['keywords'].split(', ')
            
            # Create an expandable card for each topic
            with st.expander(f"**{topic_name}** — {doc_count} paper{'s' if doc_count != 1 else ''}", expanded=idx==0):
                # Two columns: keywords on left, papers on right
                col_left, col_right = st.columns([1, 2])
                
                with col_left:
                    st.markdown("**🔑 Key Terms**")
                    
                    # Display keywords as badges with DeepSeek blue - FIXED
                    badge_html = '<div style="margin-bottom: 10px;">'
                    for i, kw in enumerate(keywords[:10]):  # Show up to 10 keywords
                        # Use DeepSeek blue variations
                        colors = ['#4f9cf9', '#2d73da', '#1a5fb4', '#0d4a8c']
                        color = colors[i % len(colors)]
                        badge_html += f'''<span style="background-color: {color}; color: white; padding: 6px 12px; margin: 4px; border-radius: 20px; font-size: 0.9em; display: inline-block; font-weight: 500;">{kw}</span>'''
                    badge_html += '</div>'
                    st.markdown(badge_html, unsafe_allow_html=True)
                    
                    # Topic stats
                    st.markdown(f"**📈 Coverage:** {doc_count} papers ({doc_count/len(papers)*100:.1f}% of total)")
                
                with col_right:
                    st.markdown("**📚 Papers in this topic**")
                    
                    # Get papers in this topic
                    papers_in_topic = services["topics"].get_papers_in_topic(
                        topic_id, 
                        papers, 
                        topic_data
                    )
                    
                    if papers_in_topic:
                        # Display papers in a clean list
                        for i, paper in enumerate(papers_in_topic[:5]):  # Show up to 5 papers
                            with st.container(border=True):
                                st.markdown(f"**{i+1}. {paper.title}**")
                                
                                # Quick metadata
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    if hasattr(paper, 'authors'):
                                        st.caption(f"👥 {paper.authors[:50]}...")
                                with col_b:
                                    if st.button("Select", key=f"select_{topic_id}_{paper.id}", 
                                                help="Select for Q&A", type="secondary"):
                                        st.session_state.selected_paper_id = paper.id
                                        st.success(f"Selected '{paper.title[:50]}...'")
                                        st.rerun()
                        
                        if len(papers_in_topic) > 5:
                            st.caption(f"... and {len(papers_in_topic) - 5} more papers")
                    else:
                        st.info("No papers found in this topic")
        
        # Topic selection for detailed view
        st.markdown("---")
        st.subheader("🔬 Explore a Specific Topic")
        
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
                
                # FIXED: Show keywords properly without duplication
                keyword_badges = ""
                for i, kw in enumerate(selected_keywords[:5]):
                    colors = ['#4f9cf9', '#2d73da', '#1a5fb4', '#0d4a8c']
                    color = colors[i % len(colors)]
                    keyword_badges += f'<span style="background-color: {color}; color: white; padding: 4px 8px; margin: 2px; border-radius: 15px; font-size: 0.85em; display: inline-block;">{kw}</span> '
                
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
    # CUSTOM CSS - NO ORANGE, BLUE SLIDERS
    # ============================================
    st.markdown("""
    <style>
    /* Remove Streamlit default padding at top */
    .stApp {
        margin-top: -60px !important;
        padding-top: 0 !important;
    }
    
    /* Main header container - NO BACKGROUND */
    .paperminer-main {
        text-align: center;
        margin: -25px 0 1rem 0;
        padding: 0;
        position: relative;
    }
    
    /* Main title - YELLOW ONLY (no orange shadow) */
    .paperminer-title {
        font-size: 3.8rem;
        font-weight: 900;
        color: #FFD700 !important;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
        padding-top: 5px;
    }
    
    /* Divider with DeepSeek blue ONLY */
    .divider {
        height: 4px;
        background: #4f9cf9 !important;
        width: 250px;
        margin: 0.3rem auto 0.5rem auto;
        border-radius: 4px;
    }
    
    /* Subtitle - DeepSeek blue */
    .paperminer-subtitle {
        font-size: 1.3rem;
        color: #2d73da !important;
        font-weight: 500;
        margin-top: 0.1rem;
    }
    
    /* SLIDER STYLING - BLUE */
    .stSlider > div > div > div {
        background-color: #4f9cf9 !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #2d73da !important;
        border-color: #2d73da !important;
    }
    
    .stSlider label {
        color: #1a5fb4 !important;
        font-weight: 600 !important;
    }
    
    /* Tab headers styling - blue */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 12px 24px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4f9cf9 !important;
        color: white !important;
        border-radius: 8px 8px 0 0;
    }
    
    /* Headers in each tab - blue */
    h1, h2, h3 {
        color: #1a5fb4 !important;
    }
    
    /* Primary buttons - DeepSeek blue */
    .stButton > button {
        background-color: #4f9cf9 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:hover {
        background-color: #2d73da !important;
        color: white !important;
        border: none !important;
    }
    
    /* Sidebar headers - blue */
    .sidebar .stMarkdown h3 {
        color: #1a5fb4 !important;
    }
    
    /* Ensure space at bottom of page */
    .main .block-container {
        padding-bottom: 4rem !important;
    }
    
    /* Footer space */
    .footer-spacer {
        height: 80px;
        width: 100%;
    }
    
    /* Metrics styling - blue */
    [data-testid="stMetric"] {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4f9cf9;
    }
    
    /* Expanders - blue */
    .streamlit-expanderHeader {
        background-color: #f0f7ff;
        border-left: 4px solid #4f9cf9;
    }
    
    /* Remove any orange text */
    * {
        color: #1a5fb4 !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #f8fafc !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # ============================================
    # VISIBLE PAGE HEADER - YELLOW TITLE ONLY
    # ============================================
    st.markdown("""
    <div class="paperminer-main">
        <h1 class="paperminer-title">📚 PAPERMINER</h1>
        <div class="divider"></div>
        <p class="paperminer-subtitle">Scientific Literature Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # ============================================
    # SIDEBAR SETTINGS WITH BLUE SLIDERS
    # ============================================
    st.sidebar.markdown("### ⚙️ Settings")
    
    st.sidebar.markdown("#### Search Settings")
    top_k = st.sidebar.slider("Number of results", 5, 50, 10, step=5)
    max_len = st.sidebar.slider("Max summary length", 64, 256, 128, step=16)
    min_len = st.sidebar.slider("Min summary length", 16, 64, 32, step=8)
    
    # Topic analysis tips
    with st.sidebar.expander("💡 How to get better topics"):
        st.markdown("""
        - **Search for 10+ papers** for clearer topics
        - Use **specific queries** (e.g., "federated learning privacy" vs "machine learning")
        - Try different **arXiv categories** to focus your search
        - Generic words (data, model, using) are automatically filtered out
        """)
    
    # About section
    with st.sidebar.expander("ℹ️ About PAPERMINER"):
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
    tab_summaries, tab_topics, tab_qa = st.tabs(["📄 Paper Summaries", "📊 Topic Insights", "❓ Paper Q&A"])
    
    # ============================================
    # SUMMARIES TAB
    # ============================================
    with tab_summaries:
        st.header("📄 Search & Summarize Papers")
        st.markdown("Find arXiv papers and generate AI-powered summaries.")
        
        # Render search controls
        search_inputs = render_search_controls()
        
        # Execute search if requested
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
                    st.session_state.selected_paper_id = None  # Reset selection
                    st.session_state.topic_data = None  # Reset topics
                    
                    # Show topic analysis readiness
                    st.success(f"✅ Found {len(results)} papers")
                    if len(results) >= 5:
                        st.info(f"✓ Enough papers for topic analysis! Go to **Topic Insights** tab.")
                    else:
                        st.warning(f"⚠ Need at least 5 papers for topic analysis. Found {len(results)}.")
                else:
                    st.info("No papers found for this query.")
        
        # Display results
        if st.session_state.search_results:
            st.markdown(f"### 📚 Showing {len(st.session_state.search_results)} Papers")
            
            for idx, paper in enumerate(st.session_state.search_results):
                render_paper_card(paper, idx, services, max_len, min_len)
        else:
            st.info("🔍 Enter a research topic above to search for papers.")
    
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
    # FOOTER SPACER
    # ============================================
    st.markdown('<div class="footer-spacer"></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()