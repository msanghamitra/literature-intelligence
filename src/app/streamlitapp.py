"""
MINIMAL UI LAYER - All business logic moved to services/
Enhanced with adaptive color palette and time-based theming
"""
from pathlib import Path
import sys
import streamlit as st
from datetime import datetime

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
# Time-based Theme Detection
# -----------------------
def get_theme_colors():
    """
    Returns color scheme based on time of day
    Morning (6am-6pm): Lighter, warmer tones
    Evening (6pm-6am): Deeper, cooler tones
    """
    current_hour = datetime.now().hour
    is_morning = 6 <= current_hour < 18
    
    if is_morning:
        # Morning theme - Brighter, energetic
        return {
            "primary": "#f3893a",      # Orange - warm and energizing
            "secondary": "#eb9830",    # Golden orange - bright accent
            "accent": "#116b98",       # Ocean blue - cool contrast
            "nature": "#4b633b",       # Forest green - grounding
            "bg_main": "#faf7f5",      # Warm off-white
            "bg_secondary": "#f5f0eb", # Light cream
            "bg_card": "#ffffff",      # Pure white for cards
            "text_primary": "#2a2a2a", # Dark gray
            "text_secondary": "#5a5a5a", # Medium gray
            "border": "#e0d5cc",       # Light warm border
            "shadow": "rgba(17, 107, 152, 0.15)", # Soft blue shadow
        }
    else:
        # Evening theme - Deeper, calming
        return {
            "primary": "#116b98",      # Ocean blue - calming primary
            "secondary": "#4b633b",    # Forest green - natural accent
            "accent": "#eb9830",       # Golden orange - warm highlight
            "nature": "#f3893a",       # Soft orange - subtle warmth
            "bg_main": "#0d1b2a",      # Deep navy
            "bg_secondary": "#1b2838", # Slate blue
            "bg_card": "#243447",      # Card background
            "text_primary": "#e8eaed", # Light gray
            "text_secondary": "#b8bec4", # Medium light gray
            "border": "#3a4a5c",       # Muted blue border
            "shadow": "rgba(235, 152, 48, 0.2)", # Warm orange glow
        }


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
    colors = get_theme_colors()

    # Search mode with styled label
    st.markdown(f"""
    <div style="margin-bottom: 0.3rem; font-weight: 600; font-size: 1.1rem; color: {colors['primary']};">
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
    
    # Semantic search explanation - styled info box
    st.markdown(f"""
    <div style='background-color: {colors['bg_card']}; padding: 10px 14px; border-radius: 8px; 
                border-left: 4px solid {colors['accent']}; margin: 10px 0 14px 0; 
                font-size: 0.9em; color: {colors['text_secondary']}; 
                box-shadow: 0 2px 4px {colors['shadow']};'>
    <strong style="color: {colors['primary']};">Semantic search</strong> finds papers based on meaning using embeddings. Better for complex topics.
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
    
    # Centered button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        submitted = st.button("Search", type="primary", use_container_width=True)
    
    # Placeholder message
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
    colors = get_theme_colors()
    
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
                
                # Display selected topic's keywords with color palette
                selected_topic_row = topics_df[topics_df['topic_id'] == selected_topic_id].iloc[0]
                selected_keywords = selected_topic_row['keywords'].split(', ')
                
                # Show keywords with new color palette
                keyword_badges = ""
                palette_colors = [colors['primary'], colors['accent'], colors['secondary'], colors['nature']]
                for i, kw in enumerate(selected_keywords[:5]):
                    color = palette_colors[i % len(palette_colors)]
                    keyword_badges += f'<span style="background-color: {color}; color: white; padding: 6px 12px; margin: 3px; border-radius: 20px; font-size: 0.88em; display: inline-block; font-weight: 500; box-shadow: 0 2px 4px {colors["shadow"]};">{kw}</span> '
                
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
# Dynamic CSS Generator
# -----------------------
def generate_css(colors):
    """Generate CSS with current theme colors"""
    is_morning = 6 <= datetime.now().hour < 18
    theme_name = "Morning" if is_morning else "Evening"
    
    return f"""
    <style>
    /* ========================================
       PAPERMINER - {theme_name} Theme
       Adaptive Color Palette
       ======================================== */
    
    /* Color Variables */
    :root {{
        --primary-color: {colors['primary']} !important;
        --secondary-color: {colors['secondary']} !important;
        --accent-color: {colors['accent']} !important;
        --nature-color: {colors['nature']} !important;
        --background-color: {colors['bg_main']} !important;
        --secondary-background-color: {colors['bg_secondary']} !important;
        --card-background: {colors['bg_card']} !important;
        --text-primary: {colors['text_primary']} !important;
        --text-secondary: {colors['text_secondary']} !important;
        --border-color: {colors['border']} !important;
        --shadow-color: {colors['shadow']} !important;
    }}
    
    /* ========== MAIN LAYOUT ========== */
    .stApp {{
        background-color: {colors['bg_main']} !important;
        margin-top: -60px !important;
        padding-top: 0 !important;
    }}
    
    .main .block-container {{
        background-color: {colors['bg_main']} !important;
        padding-top: 2rem !important;
    }}
    
    /* ========== HEADER ========== */
    .paperminer-main {{
        text-align: center;
        margin: -35px 0 0.5rem 0 !important;
        padding: 25px;
        background: transparent !important;
        position: relative;
    }}
    
    /* Title with emoji in original color and text in #116b98 */
    .paperminer-title {{
        font-size: 3.8rem;
        font-weight: 900;
        letter-spacing: 2px;
        margin-bottom: 0 !important;
        padding-top: 5px;
        line-height: 1.1;
    }}
    
    .paperminer-title .emoji {{
        color: inherit !important;
        filter: none !important;
    }}
    
    .paperminer-title .title-text {{
        color: #116b98 !important;
    }}
    
    .divider {{
        height: 4px;
        background: linear-gradient(90deg, {colors['accent']}, #116b98, {colors['secondary']}, {colors['nature']}) !important;
        width: 320px;
        margin: 0 auto 0.4rem auto !important;
        border-radius: 4px;
        box-shadow: 0 2px 8px {colors['shadow']};
    }}
    
    .paperminer-subtitle {{
        font-size: 1.3rem;
        color: {colors['primary']} !important;
        font-weight: 500;
        margin-top: 0 !important;
        opacity: 0.95;
        text-shadow: 0 0 10px {colors['shadow']};
    }}
    
    /* ========== TABS - Lighter shades between green and blue ========== */
    .stTabs [data-baseweb="tab"] {{
        font-size: 1.1rem;
        font-weight: 600;
        padding: 14px 28px;
        margin: 0 4px;
        border-radius: 10px 10px 0 0;
        transition: all 0.3s ease;
        border: 2px solid transparent;
    }}
    
    /* Tab 1 - Teal/Cyan (lighter blue-green) */
    .stTabs [data-baseweb="tab"]:first-child:not([aria-selected="true"]) {{
        background: linear-gradient(135deg, #2a9d8f, #26a3a3) !important;
        color: white !important;
        border-bottom: none;
    }}
    
    .stTabs [data-baseweb="tab"]:first-child:not([aria-selected="true"]):hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['shadow']};
    }}
    
    /* Tab 2 - Sea Green (middle range) */
    .stTabs [data-baseweb="tab"]:nth-child(2):not([aria-selected="true"]) {{
        background: linear-gradient(135deg, #3a9679, #2d8a72) !important;
        color: white !important;
        border-bottom: none;
    }}
    
    .stTabs [data-baseweb="tab"]:nth-child(2):not([aria-selected="true"]):hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['shadow']};
    }}
    
    /* Tab 3 - Forest Teal */
    .stTabs [data-baseweb="tab"]:nth-child(3):not([aria-selected="true"]) {{
        background: linear-gradient(135deg, #4a8565, #3d7658) !important;
        color: white !important;
        border-bottom: none;
    }}
    
    .stTabs [data-baseweb="tab"]:nth-child(3):not([aria-selected="true"]):hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px {colors['shadow']};
    }}
    
    /* Selected tab - Brighter teal */
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, #2a9d8f, #26b3a3) !important;
        color: white !important;
        font-weight: 700;
        border: 2px solid #2a9d8f;
        border-bottom: none;
        box-shadow: 0 0 20px rgba(42, 157, 143, 0.5), 0 4px 12px rgba(0,0,0,0.3);
        transform: translateY(-3px);
    }}
    
    /* ========== HEADERS ========== */
    h1, h2, h3, h4 {{
        color: {colors['primary']} !important;
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
        text-shadow: 0 0 10px {colors['shadow']};
    }}
    
    /* ========== RADIO BUTTONS - Circular style ========== */
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"] > div,
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"] > div > div,
    div[data-baseweb="radio"] > div,
    div[data-baseweb="radio"] > div > div {{
        border-color: {colors['nature']} !important;
        border-radius: 50% !important;
        width: 18px !important;
        height: 18px !important;
    }}
    
    /* Inner circle for checked state */
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"][aria-checked="true"] > div > div,
    div[data-baseweb="radio"] input:checked ~ div > div,
    div[data-baseweb="radio"] input:checked + div > div {{
        background-color: {colors['primary']} !important;
        border-color: {colors['primary']} !important;
        border-radius: 50% !important;
        width: 10px !important;
        height: 10px !important;
        box-shadow: 0 0 10px {colors['shadow']};
    }}
    
    /* Outer circle for checked state */
    [data-testid="stRadio"] [role="radiogroup"] [role="radio"][aria-checked="true"] > div,
    div[data-baseweb="radio"] input:checked ~ div,
    div[data-baseweb="radio"] input:checked + div {{
        border-color: {colors['primary']} !important;
        border-radius: 50% !important;
        border-width: 2px !important;
    }}
    
    .stRadio label,
    [data-testid="stRadio"] label {{
        color: {colors['primary']} !important;
        font-weight: 500;
    }}
    
    /* ========== BUTTONS ========== */
    button[kind="primary"],
    button[data-testid="baseButton-primary"],
    .stButton > button,
    .stDownloadButton > button,
    div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, {colors['primary']}, {colors['secondary']}) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px {colors['shadow']};
    }}
    
    button[kind="primary"]:hover,
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px {colors['shadow']};
        background: linear-gradient(135deg, {colors['secondary']}, {colors['accent']}) !important;
    }}
    
    button[kind="secondary"],
    button[data-testid="baseButton-secondary"] {{
        background: linear-gradient(135deg, {colors['nature']}, {colors['accent']}) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px {colors['shadow']};
    }}
    
    button[kind="secondary"]:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px {colors['shadow']};
    }}
    
    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {{
        background-color: {colors['bg_secondary']} !important;
        border-right: 2px solid {colors['border']};
    }}
    
    /* Sidebar main headers - Forest Green #4b633b */
    .sidebar .stMarkdown h3,
    [data-testid="stSidebar"] h3 {{
        color: #4b633b !important;
        border-bottom: 2px solid #4b633b;
        padding-bottom: 0.6rem;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(75, 99, 59, 0.3);
    }}
    
    /* Sidebar subheaders - Forest Green */
    .sidebar .stMarkdown h4,
    [data-testid="stSidebar"] h4 {{
        color: #4b633b !important;
        font-weight: 600;
    }}
    
    /* ========== SLIDERS - NUCLEAR OPTION ========== */
    /* Kill ALL backgrounds in slider area */
    section[data-testid="stSidebar"] .stSlider,
    section[data-testid="stSidebar"] .stSlider *,
    section[data-testid="stSidebar"] [data-testid="stSlider"],
    section[data-testid="stSidebar"] [data-testid="stSlider"] *,
    [data-testid="stSidebar"] .element-container,
    [data-testid="stSidebar"] .stSlider > div,
    [data-testid="stSidebar"] .stSlider > div > div,
    [data-testid="stSidebar"] .stSlider > div > div > div,
    [data-testid="stSidebar"] .stSlider > div > div > div > div,
    [data-testid="stSidebar"] .stSlider div[data-baseweb="slider"],
    [data-testid="stSidebar"] div[data-baseweb="slider"] *,
    .stSlider,
    .stSlider *,
    [data-testid="stSlider"],
    [data-testid="stSlider"] * {{
        background-color: transparent !important;
        background: none !important;
        background-image: none !important;
    }}
    
    /* Labels */
    [data-testid="stSidebar"] .stSlider label,
    .stSlider label {{
        color: {colors['primary']} !important;
        font-weight: 600 !important;
        background: transparent !important;
    }}
    
    /* Track container - transparent */
    [data-testid="stSidebar"] div[data-baseweb="slider"],
    div[data-baseweb="slider"] {{
        background: transparent !important;
    }}
    
    /* Gray unfilled track */
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:nth-child(1),
    div[data-baseweb="slider"] > div:nth-child(1) {{
        background: transparent !important;
    }}
    
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(1),
    div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(1) {{
        background-color: #d0d0d0 !important;
        height: 4px !important;
        border-radius: 2px !important;
    }}
    
    /* Green filled track (progress bar) */
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(2),
    div[data-baseweb="slider"] > div:nth-child(1) > div:nth-child(2) {{
        background-color: #4b633b !important;
        background: #4b633b !important;
        height: 4px !important;
        border-radius: 2px !important;
    }}
    
    /* Thumb container - transparent */
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:nth-child(2),
    div[data-baseweb="slider"] > div:nth-child(2) {{
        background: transparent !important;
    }}
    
    /* Circular orange thumb */
    [data-testid="stSidebar"] div[data-baseweb="slider"] > div:nth-child(2) > div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"],
    div[data-baseweb="slider"] > div:nth-child(2) > div,
    div[data-baseweb="slider"] div[role="slider"] {{
        background-color: #eb9830 !important;
        background: #eb9830 !important;
        border: 3px solid #4b633b !important;
        border-radius: 50% !important;
        width: 20px !important;
        height: 20px !important;
        box-shadow: 0 2px 6px rgba(235, 152, 48, 0.5) !important;
    }}
    
    /* Kill inner thumb elements */
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] > div,
    [data-testid="stSidebar"] div[data-baseweb="slider"] div[role="slider"] > div > div,
    div[data-baseweb="slider"] div[role="slider"] > div,
    div[data-baseweb="slider"] div[role="slider"] > div > div {{
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }}
    
    /* Hide value display box */
    [data-testid="stSidebar"] .stSlider div[data-baseweb="tooltip"],
    .stSlider div[data-baseweb="tooltip"] {{
        display: none !important;
    }}
    
    /* ========== CHECKBOXES ========== */
    .stCheckbox [data-baseweb="checkbox"] div:first-child,
    [data-testid="stCheckbox"] [role="checkbox"] {{
        border-color: {colors['primary']} !important;
    }}
    
    .stCheckbox input:checked + div > div:first-child,
    [data-testid="stCheckbox"] input:checked ~ div {{
        background-color: {colors['primary']} !important;
        border-color: {colors['primary']} !important;
        box-shadow: 0 0 8px {colors['shadow']};
    }}
    
    /* ========== EXPANDERS ========== */
    .sidebar .streamlit-expanderHeader,
    .streamlit-expanderHeader {{
        color: {colors['primary']} !important;
        font-weight: 700 !important;
        background-color: {colors['bg_card']} !important;
        border-radius: 6px;
        border: 1px solid {colors['border']};
    }}
    
    .streamlit-expanderContent {{
        background-color: {colors['bg_card']} !important;
        border: 1px solid {colors['border']};
        border-radius: 0 0 8px 8px;
    }}
    
    /* ========== TEXT COLORS ========== */
    * {{
        color: {colors['text_primary']} !important;
    }}
    
    p, span, div {{
        color: {colors['text_primary']} !important;
    }}
    
    /* ========== INPUT FIELDS ========== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {{
        background-color: {colors['bg_card']} !important;
        color: {colors['text_primary']} !important;
        border: 2px solid {colors['border']} !important;
        border-radius: 8px;
    }}
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {colors['primary']} !important;
        box-shadow: 0 0 10px {colors['shadow']};
    }}
    
    /* ========== METRICS ========== */
    [data-testid="stMetricValue"] {{
        color: {colors['primary']} !important;
        font-weight: 700;
    }}
    
    [data-testid="stMetricLabel"] {{
        color: {colors['secondary']} !important;
    }}
    
    /* ========== ALERTS ========== */
    .stAlert {{
        background-color: {colors['bg_card']} !important;
        border-left: 4px solid {colors['accent']} !important;
        color: {colors['text_primary']} !important;
        border-radius: 6px;
    }}
    
    /* ========== CONTAINERS ========== */
    .element-container,
    [data-testid="stExpander"],
    [data-testid="stVerticalBlock"] {{
        background-color: transparent !important;
    }}
    </style>
    
    <script>
    // Force remove slider backgrounds
    setTimeout(function() {{
        const sliders = document.querySelectorAll('.stSlider, [data-testid="stSlider"]');
        sliders.forEach(function(slider) {{
            const allDivs = slider.querySelectorAll('div');
            allDivs.forEach(function(div) {{
                if (!div.hasAttribute('role')) {{
                    div.style.background = 'transparent';
                    div.style.backgroundColor = 'transparent';
                }}
            }});
        }});
    }}, 100);
    </script>
    """


# -----------------------
# Main App
# -----------------------
def main():
    """Main application entry point"""
    # Initialize
    init_session_state()
    services = init_services()
    
    # Get theme colors
    colors = get_theme_colors()
    current_hour = datetime.now().hour
    is_morning = 6 <= current_hour < 18
    theme_emoji = "☀️" if is_morning else "🌙"
    theme_name = "Morning" if is_morning else "Evening"
    
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
    # DYNAMIC CSS
    # ============================================
    # Add cache buster to force CSS reload
    import random
    css_version = random.randint(1000, 9999)
    st.markdown(f'<div id="css-reload-{css_version}"></div>', unsafe_allow_html=True)
    st.markdown(generate_css(colors), unsafe_allow_html=True)

    # ============================================
    # PAGE HEADER - Emoji in original color, title in #116b98
    # ============================================
    st.markdown("""
    <div class="paperminer-main">
        <h1 class="paperminer-title">
            <span class="emoji">🔍</span> 
            <span class="title-text">PAPERMINER</span>
        </h1>
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
    # SUMMARIES TAB
    # ============================================
    with tab_summaries:
        # Compact header
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <h2 style="margin: 0; padding: 0;">📄 Search & Paper Summaries</h2>
            <p style="margin: 0.1rem 0 0 0; color: {colors['text_secondary']}; font-size: 0.9rem;">
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
