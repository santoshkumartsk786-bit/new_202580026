"""
RAG-based Movie Review Question-Answering System
TRUE GENERATION - Synthesizes insights from retrieved evidence
Uses FLAN-T5-base for meaningful answer generation
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import re

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Movie Review RAG System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .answer-box {
        background-color: #f0f7ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .positive-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .negative-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_all_resources():
    """Load encoder, generator, FAISS index, and metadata"""
    
    try:
        # Load sentence transformer for retrieval
        encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Load FLAN-T5-BASE for TRUE generation (not small!)
        # This is 3x larger and produces much better answers
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = generator.to(device)
        
        # Load FAISS index
        if not os.path.exists("models/faiss_index.bin"):
            return None, None, None, None, None, None, "faiss_missing"
        
        index = faiss.read_index("models/faiss_index.bin")
        
        # Load metadata
        if not os.path.exists("models/review_metadata.pkl"):
            return None, None, None, None, None, None, "metadata_missing"
        
        with open('models/review_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load build report if exists
        build_report = None
        if os.path.exists('models/build_report.json'):
            with open('models/build_report.json', 'r') as f:
                build_report = json.load(f)
        
        return encoder, tokenizer, generator, device, index, metadata, build_report
    
    except Exception as e:
        return None, None, None, None, None, None, str(e)

# ============================================================================
# QUERY VALIDATION
# ============================================================================

def is_meaningful_query(query):
    """Check if query is meaningful and not gibberish"""
    if not query or not isinstance(query, str):
        return False, "Empty query"
    
    query = query.strip()
    
    if len(query) < 5:
        return False, "Query too short"
    
    alpha_chars = sum(c.isalpha() for c in query)
    if alpha_chars < 3:
        return False, "Query contains insufficient letters"
    
    gibberish_patterns = [
        r'^[^aeiou\s]{8,}$',
        r'^(.)\1{5,}',
        r'^[^a-zA-Z0-9\s]{5,}',
    ]
    
    for pattern in gibberish_patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return False, "Query appears to be nonsense"
    
    words = query.split()
    if len(words) < 2:
        return False, "Query needs at least 2 words"
    
    if len(set(words)) < len(words) * 0.5:
        return False, "Query has too much repetition"
    
    return True, "Valid query"

# ============================================================================
# RETRIEVAL SYSTEM
# ============================================================================

def retrieve_reviews(query, encoder, index, metadata, top_k=7, filter_sentiment=None):
    """
    Retrieve relevant reviews using semantic search
    
    Returns:
        (results, quality_metrics)
    """
    # Encode query
    query_embedding = encoder.encode([query], convert_to_numpy=True)
    
    # Search FAISS
    search_k = min(top_k * 3, len(metadata['review_ids']))
    distances, indices = index.search(query_embedding.astype('float32'), search_k)
    
    results = []
    MIN_SIMILARITY = 0.30  # Relaxed for more context
    
    for idx, distance in zip(indices[0], distances[0]):
        if idx >= len(metadata['sentiments']):
            continue
        
        # Convert distance to similarity
        similarity = float(1 / (1 + distance))
        
        if similarity < MIN_SIMILARITY:
            continue
            
        sentiment = metadata['sentiments'][idx]
        
        if filter_sentiment and sentiment != filter_sentiment:
            continue
        
        result = {
            'review_id': metadata['review_ids'][idx],
            'text': metadata['texts'][idx],
            'sentiment': sentiment,
            'similarity_score': similarity
        }
        
        if metadata['ratings'] and idx < len(metadata['ratings']):
            result['rating'] = metadata['ratings'][idx]
        
        results.append(result)
        
        if len(results) >= top_k:
            break
    
    # Quality metrics
    if not results:
        quality = {
            'is_sufficient': False,
            'result_count': 0,
            'avg_similarity': 0.0,
            'top_similarity': 0.0
        }
    else:
        quality = {
            'is_sufficient': len(results) >= 3 and results[0]['similarity_score'] >= 0.40,
            'result_count': len(results),
            'avg_similarity': sum(r['similarity_score'] for r in results) / len(results),
            'top_similarity': results[0]['similarity_score']
        }
    
    return results, quality


def analyze_sentiment_distribution(results):
    """Analyze sentiment distribution in results"""
    if not results:
        return {}
    
    sentiments = [r['sentiment'] for r in results]
    pos_count = sentiments.count('positive')
    neg_count = sentiments.count('negative')
    total = len(sentiments)
    
    return {
        'positive_count': pos_count,
        'negative_count': neg_count,
        'positive_pct': (pos_count / total * 100) if total > 0 else 0,
        'negative_pct': (neg_count / total * 100) if total > 0 else 0,
        'dominant': 'positive' if pos_count > neg_count else 'negative'
    }

# ============================================================================
# TRUE ANSWER GENERATION
# ============================================================================

def generate_comprehensive_answer(query, results, quality_metrics, tokenizer, generator, device):
    """
    Generate TRUE answers that synthesize insights from evidence
    Uses FLAN-T5-base with enhanced prompting for analytical responses
    
    This is REAL generation - not just extraction or citation listing
    """
    
    if not quality_metrics['is_sufficient']:
        return "I couldn't find enough relevant reviews to answer your question. Please try rephrasing or asking about a different aspect of movies."
    
    # Build rich evidence context
    evidence_texts = []
    for i, result in enumerate(results[:6], 1):  # Use more reviews for better synthesis
        # Include full context (not truncated)
        text = result['text'][:600]  # Longer snippets for better understanding
        sentiment = result['sentiment']
        rating_info = f" (Rating: {result['rating']}/10)" if 'rating' in result else ""
        
        evidence_texts.append(
            f"Review {i} [{sentiment.upper()}{rating_info}]: {text}"
        )
    
    evidence_block = "\n\n".join(evidence_texts)
    
    # Sentiment analysis
    sent_dist = analyze_sentiment_distribution(results)
    sentiment_summary = f"{sent_dist['positive_count']} positive, {sent_dist['negative_count']} negative reviews"
    
    # Create prompt for SYNTHESIS (not extraction)
    prompt = f"""You are a movie review analyst. Based on the reviews below, provide a comprehensive, synthesized answer to the question.

Question: {query}

Reviews analyzed ({sentiment_summary}):
{evidence_block}

Instructions:
1. SYNTHESIZE insights across multiple reviews - don't just list what each review says
2. Identify common themes, patterns, and contradictions
3. Explain the reasoning behind sentiments
4. Provide a cohesive narrative answer
5. Be analytical and insightful, not just descriptive
6. Write in complete paragraphs with natural flow
7. Connect ideas across different reviews

Provide your analytical answer:"""

    # Generate with settings optimized for synthesis
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=1536,  # More context for synthesis
        truncation=True
    ).to(device)
    
    with torch.no_grad():
        outputs = generator.generate(
            **inputs,
            max_length=300,  # Longer for comprehensive answers
            min_length=80,   # Ensure substantial response
            num_beams=5,     # Better quality
            temperature=0.85,  # More creative synthesis
            do_sample=True,  # Enable sampling for variety
            top_p=0.92,      # Nucleus sampling
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Add context about evidence quality
    confidence_note = ""
    if quality_metrics['avg_similarity'] >= 0.55:
        confidence_note = "high confidence"
    elif quality_metrics['avg_similarity'] >= 0.45:
        confidence_note = "good confidence"
    else:
        confidence_note = "moderate confidence"
    
    final_answer = f"{answer}\n\n*Analysis based on {quality_metrics['result_count']} reviews ({confidence_note}, avg relevance: {quality_metrics['avg_similarity']:.2f})*"
    
    return final_answer

# ============================================================================
# COMPLETE RAG PIPELINE
# ============================================================================

def rag_pipeline(query, encoder, index, metadata, tokenizer, generator, device,
                 top_k=7, filter_sentiment=None):
    """
    Complete RAG pipeline: Retrieve → Analyze → Generate
    """
    # Step 1: Validate query
    is_valid, validation_reason = is_meaningful_query(query)
    
    if not is_valid:
        return (
            "Please provide a meaningful question about movies or reviews.",
            [],
            {'is_sufficient': False, 'result_count': 0, 'avg_similarity': 0.0, 'top_similarity': 0.0},
            'invalid_query'
        )
    
    # Step 2: Retrieve relevant reviews
    results, quality_metrics = retrieve_reviews(
        query, encoder, index, metadata, top_k, filter_sentiment
    )
    
    # Step 3: Generate comprehensive answer
    answer = generate_comprehensive_answer(
        query, results, quality_metrics, tokenizer, generator, device
    )
    
    status = 'success' if quality_metrics['is_sufficient'] else 'insufficient_evidence'
    
    return answer, results, quality_metrics, status

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.markdown('<p class="main-header">🎬 Movie Review Question-Answering System</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        <p>Ask questions about movies and get AI-generated answers based on real reviews</p>
        <p style="font-size: 0.9em; color: #999;">✨ Powered by FLAN-T5-base for true answer synthesis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load resources
    with st.spinner("🔄 Loading AI models and review database..."):
        encoder, tokenizer, generator, device, index, metadata, error_or_report = load_all_resources()
        
        if encoder is None:
            if error_or_report == "faiss_missing":
                st.error("❌ Missing FAISS index. Run `python build_index.py` first.")
            elif error_or_report == "metadata_missing":
                st.error("❌ Missing metadata. Run `python build_index.py` first.")
            else:
                st.error(f"❌ Error: {error_or_report}")
            st.stop()
        
        total_reviews = len(metadata['review_ids'])
        build_report = error_or_report if isinstance(error_or_report, dict) else None
    
    # Show dataset info
    if build_report:
        with st.expander("ℹ️ Dataset Information", expanded=False):
            st.markdown(f"""
            <div class="info-box">
                📊 Total Reviews: {build_report.get('total_reviews', 'N/A'):,}<br>
                ✅ Positive: {build_report.get('positive_reviews', 'N/A'):,} ({build_report.get('positive_percentage', 0):.1f}%)<br>
                ❌ Negative: {build_report.get('negative_reviews', 'N/A'):,} ({build_report.get('negative_percentage', 0):.1f}%)<br>
                🤖 Generator: FLAN-T5-base (250M parameters)<br>
                💾 Index Size: {build_report.get('index_size_mb', 0):.2f} MB
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Reviews to analyze",
            min_value=5,
            max_value=10,
            value=7,
            help="More reviews = more comprehensive answers"
        )
        
        unique_sentiments = set(metadata['sentiments'])
        sentiment_options = ["All"] + sorted([s.title() for s in unique_sentiments if s])
        
        filter_sentiment = st.selectbox("Filter by sentiment", sentiment_options)
        filter_val = None if filter_sentiment == "All" else filter_sentiment.lower()
        
        show_sources = st.checkbox("Show source reviews", value=True)
        show_chart = st.checkbox("Show sentiment chart", value=True)
        
        st.divider()
        st.header("📊 Database")
        st.metric("Total Reviews", f"{total_reviews:,}")
        
        sentiment_counts = {}
        for s in metadata['sentiments']:
            sentiment_counts[s] = sentiment_counts.get(s, 0) + 1
        
        for sentiment, count in sorted(sentiment_counts.items()):
            if sentiment:
                st.metric(sentiment.title(), f"{count:,}")
        
        st.caption(f"Device: {device.upper()}")
        
        st.divider()
        st.header("💡 Example Questions")
        examples = [
            "Why do some movies receive negative reviews?",
            "What makes a horror movie effective?",
            "How do viewers respond to poor acting?",
            "What elements make a movie entertaining?",
            "Why do some comedies fail to be funny?",
            "What creates emotional impact in films?",
            "How important is cinematography to viewers?"
        ]
        
        for ex in examples:
            if st.button(ex, key=ex, use_container_width=True):
                st.session_state.example_query = ex
    
    # Main interface
    st.header("🔍 Ask Your Question")
    
    default_query = st.session_state.get('example_query', '')
    if 'example_query' in st.session_state:
        del st.session_state.example_query
    
    query = st.text_input(
        "Enter your question:",
        value=default_query,
        placeholder="e.g., Why do horror movies scare audiences?",
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("🔎 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.rerun()
    
    # Process query
    if analyze_btn and query:
        with st.spinner("🤔 Retrieving evidence and generating answer..."):
            try:
                start = datetime.now()
                
                answer, results, quality_metrics, status = rag_pipeline(
                    query=query,
                    encoder=encoder,
                    index=index,
                    metadata=metadata,
                    tokenizer=tokenizer,
                    generator=generator,
                    device=device,
                    top_k=top_k,
                    filter_sentiment=filter_val
                )
                
                elapsed = (datetime.now() - start).total_seconds()
                
                st.divider()
                
                if status == 'invalid_query':
                    st.warning("⚠️ " + answer)
                    
                elif status == 'insufficient_evidence':
                    st.warning("⚠️ " + answer)
                    st.info(f"""
                    **Found {quality_metrics['result_count']} reviews, but they weren't relevant enough.**
                    
                    Try:
                    - More general questions
                    - Different keywords
                    - Remove sentiment filter
                    """)
                    
                else:  # success
                    st.success("✅ Answer generated from review analysis")
                    
                    # Metrics
                    cols = st.columns(4)
                    cols[0].metric("Reviews Analyzed", quality_metrics['result_count'])
                    cols[1].metric("Avg Relevance", f"{quality_metrics['avg_similarity']:.2f}")
                    cols[2].metric("Top Match", f"{quality_metrics['top_similarity']:.2f}")
                    cols[3].metric("Time", f"{elapsed:.1f}s")
                    
                    # Sentiment distribution
                    if results:
                        sent_dist = analyze_sentiment_distribution(results)
                        st.divider()
                        cols2 = st.columns(3)
                        cols2[0].metric("Total Reviews", len(results))
                        cols2[1].metric("Positive", sent_dist['positive_count'])
                        cols2[2].metric("Negative", sent_dist['negative_count'])
                    
                    # Display answer
                    st.subheader("💡 Generated Answer")
                    st.markdown(f"""
                    <div class="answer-box">
                        {answer}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Chart
                    if show_chart and results:
                        st.subheader("📊 Evidence Sentiment")
                        
                        chart_data = []
                        chart_colors = []
                        chart_labels = []
                        
                        if sent_dist['positive_count'] > 0:
                            chart_data.append(sent_dist['positive_count'])
                            chart_colors.append('#28a745')
                            chart_labels.append('Positive')
                        
                        if sent_dist['negative_count'] > 0:
                            chart_data.append(sent_dist['negative_count'])
                            chart_colors.append('#dc3545')
                            chart_labels.append('Negative')
                        
                        if chart_data:
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=chart_labels,
                                    y=chart_data,
                                    marker_color=chart_colors,
                                    text=chart_data,
                                    textposition='auto'
                                )
                            ])
                            fig.update_layout(
                                height=300,
                                margin=dict(l=20, r=20, t=20, b=20),
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Sources
                    if show_sources and results:
                        st.subheader("📚 Source Reviews")
                        for i, result in enumerate(results[:5], 1):
                            sentiment = result['sentiment']
                            badge = "positive-badge" if sentiment == 'positive' else "negative-badge"
                            rating_txt = f" | ⭐ {result['rating']}/10" if 'rating' in result else ""
                            
                            with st.expander(
                                f"Review #{result['review_id']} - {sentiment.title()} "
                                f"(Relevance: {result['similarity_score']:.2f}){rating_txt}",
                                expanded=(i <= 2)
                            ):
                                st.markdown(f'<span class="{badge}">{sentiment.upper()}</span>', 
                                          unsafe_allow_html=True)
                                st.caption(f"Relevance: {result['similarity_score']:.3f}")
                                if 'rating' in result:
                                    st.caption(f"Rating: ⭐ {result['rating']}/10")
                                st.write(result['text'][:600])
                    
                    # Export
                    st.divider()
                    st.subheader("💾 Export")
                    
                    export_data = {
                        'query': query,
                        'answer': answer,
                        'quality_metrics': quality_metrics,
                        'sentiment_distribution': sent_dist if results else {},
                        'processing_time': elapsed,
                        'sources': results[:5]
                    }
                    
                    col1, col2 = st.columns(2)
                    col1.download_button(
                        "📥 Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"qa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                    
                    if results:
                        sources_df = pd.DataFrame([
                            {
                                'Review_ID': r['review_id'],
                                'Sentiment': r['sentiment'],
                                'Relevance': r['similarity_score'],
                                'Rating': r.get('rating', 'N/A'),
                                'Text': r['text'][:200]
                            }
                            for r in results[:5]
                        ])
                        col2.download_button(
                            "📥 Download CSV",
                            data=sources_df.to_csv(index=False),
                            file_name=f"sources_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                with st.expander("🔍 Details"):
                    st.exception(e)
    
    elif analyze_btn:
        st.warning("⚠️ Please enter a question first!")
    
    # Footer
    st.divider()
    st.caption("🎬 RAG Question-Answering System | FLAN-T5-base + Sentence Transformers")
    st.caption("This system generates comprehensive answers by synthesizing insights from multiple reviews")

if __name__ == "__main__":
    main()
