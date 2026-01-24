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
    st.markdown("### Search")
    
    search_mode = st.radio(
        "Search mode",
        ["Keyword (fast)", "Semantic (embeddings)"],
        index=0,
        horizontal=True
    )
    
    query = st.text_input(
        "Topic",
        placeholder="e.g. machine learning, diffusion models, federated learning…",
        label_visibility="collapsed"
    )
    
    arxiv_category = st.text_input(
        "Optional arXiv category (e.g., cs.LG, cs.CL, stat.ML). Leave empty for broad search.",
        value="",
    )
    
    submitted = st.button("🔍 Search", type="primary")
    
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
    st.header("Topics (from current search results)")
    
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
                st.success(f"Generated {topic_data.get('num_topics', 0)} topics!")
            else:
                st.warning(topic_data.get("error", "Could not generate topics"))
                st.session_state.topic_data = {"error": topic_data.get("error")}
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
        
        st.subheader("Topic list")
        st.dataframe(topics_df, width="stretch")
        
        # Topic selection UI
        if "topic_id" in topics_df.columns:
            topic_ids = topics_df["topic_id"].tolist()
            topic_names = [f"Topic {tid} ({topics_df.loc[topics_df['topic_id'] == tid, 'doc_count'].iloc[0]} papers)" 
                          for tid in topic_ids]
            
            selected_topic_name = st.selectbox("Browse papers by topic", topic_names)
            
            # Extract topic ID from selection
            selected_topic_id = topic_ids[topic_names.index(selected_topic_name)]
            
            # Get papers in selected topic using service
            papers_in_topic = services["topics"].get_papers_in_topic(
                selected_topic_id, 
                papers, 
                topic_data
            )
            
            if papers_in_topic:
                st.write(f"**Papers in {selected_topic_name}:**")
                
                # Paper selection within topic (UI only)
                paper_titles = [p.title for p in papers_in_topic]
                chosen_title = st.selectbox("Select a paper", paper_titles, key="topic_paper_select")
                
                if chosen_title:
                    chosen_paper = next(p for p in papers_in_topic if p.title == chosen_title)
                    if st.button("Use this paper for Q&A", key="use_for_qa"):
                        st.session_state.selected_paper_id = chosen_paper.id
                        st.success("Selected. Go to the Q&A tab.")
                        
                    # Show paper details
                    with st.expander("Paper details"):
                        st.write(f"**Title:** {chosen_paper.title}")
                        st.write(f"**Authors:** {chosen_paper.authors}")
                        st.write(f"**Abstract:** {chosen_paper.abstract[:300]}...")
            else:
                st.info("No papers found in this topic.")
    else:
        st.info("No topics generated. Try a different search.")


def render_qa_tab(services):
    """Render Q&A tab"""
    st.header("Q&A (from full paper PDF)")
    
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
    question = st.text_input("Ask a question about this paper (full PDF)")
    
    if st.button("🤖 Answer"):
        if not question or not question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving relevant sections + answering…"):
                answer = services["qa"].answer_question(selected_paper, question)
            
            if answer.get("error"):
                st.warning(answer["error"])
            else:
                st.markdown("### Answer")
                st.write(answer.get("answer", ""))
                
                # Show metrics
                if "score" in answer:
                    st.caption(f"QA confidence: {answer['score']:.3f}")
                if "similarity" in answer:
                    st.caption(f"Best chunk similarity: {answer['similarity']:.3f}")
                if "page" in answer:
                    st.caption(f"Page: {answer['page']}")
                
                # Show context if available
                if answer.get("context_snippet"):
                    with st.expander("Context snippet (from PDF)"):
                        st.write(answer["context_snippet"])
                
                # Debug info
                if answer.get("chunks_used"):
                    with st.expander("Chunks used (debug)"):
                        st.json(answer["chunks_used"])


# -----------------------
# Main App
# -----------------------
def main():
    """Main application entry point"""
    # Initialize
    init_session_state()
    services = init_services()
    
    # Page configuration
    st.set_page_config(
        page_title="Research Librarian",
        layout="wide",
        page_icon="📚"
    )
    
    # Sidebar settings
    st.sidebar.title("Settings")
    top_k = st.sidebar.slider("Number of results", 5, 50, 10, step=5)
    max_len = st.sidebar.slider("Max summary length", 64, 256, 128, step=16)
    min_len = st.sidebar.slider("Min summary length", 16, 64, 32, step=8)
    
    # Main tabs
    tab_summaries, tab_topics, tab_qa = st.tabs(["Summaries", "Topic Insights", "Q&A"])
    
    # -----------------------
    # Summaries Tab
    # -----------------------
    with tab_summaries:
        st.header("Search papers & view summaries")
        
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
                    
                    st.success(f"Found {len(results)} papers")
                else:
                    st.info("No papers found for this query.")
        
        # Display results
        if st.session_state.search_results:
            st.write(f"Showing **{len(st.session_state.search_results)}** papers.")
            
            for idx, paper in enumerate(st.session_state.search_results):
                render_paper_card(paper, idx, services, max_len, min_len)
        else:
            st.info("Search for a topic to see results.")
    
    # -----------------------
    # Topics Tab
    # -----------------------
    with tab_topics:
        render_topics_tab(services)
    
    # -----------------------
    # Q&A Tab
    # -----------------------
    with tab_qa:
        render_qa_tab(services)


if __name__ == "__main__":
    main()