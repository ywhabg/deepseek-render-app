import os
import re
import json
import time
import hashlib
from datetime import datetime
from collections import deque
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from openai import OpenAI
from flask import Flask, render_template, request, jsonify, session, flash, redirect, url_for
from flask_session import Session
import secrets

# =========================
# Configuration
# =========================

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
Session(app)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

REQUEST_TIMEOUT = 20
CRAWL_DELAY_SECONDS = 0.8
MAX_PAGES_DEFAULT = 30
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0

CHUNK_SIZE_CHARS = 6000
CHUNK_OVERLAP_CHARS = 500
MAX_CHUNKS_PER_PAGE = 8
FINAL_SUMMARY_WORD_LIMIT = 150

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid"
}

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".pdf", ".zip", ".rar", ".7z", ".tar", ".gz",
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".mp4", ".mp3", ".avi", ".mov", ".wmv",
    ".css", ".js", ".json", ".xml"
}

SKIP_KEYWORDS = {
    "login", "logout", "signin", "signup", "register",
    "cart", "checkout", "account", "profile",
    "search?", "/search", "wp-admin", "admin"
}

LOW_VALUE_PATTERNS = [
    "privacy policy",
    "cookie policy",
    "terms of use",
    "terms and conditions",
    "page not found",
    "404",
    "access denied",
    "sign in",
    "log in",
]

# =========================
# HTTP session
# =========================

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# =========================
# Utility functions
# =========================

def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY not found. Put it in your .env file or environment variables."
        )

    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    clean_query_pairs = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if key.lower() not in TRACKING_PARAMS:
            clean_query_pairs.append((key, value))

    clean_query = urlencode(clean_query_pairs, doseq=True)
    normalized_path = parsed.path.rstrip("/") if parsed.path != "/" else "/"

    normalized = parsed._replace(
        path=normalized_path,
        query=clean_query,
        fragment=""
    )
    return urlunparse(normalized)

def is_same_domain(base_url: str, target_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(target_url).netloc

def should_skip_url(url: str) -> bool:
    lower_url = url.lower()

    if lower_url.startswith("mailto:"):
        return True
    if lower_url.startswith("javascript:"):
        return True
    if lower_url.startswith("tel:"):
        return True

    for ext in SKIP_EXTENSIONS:
        if lower_url.endswith(ext):
            return True

    for keyword in SKIP_KEYWORDS:
        if keyword in lower_url:
            return True

    return False

def is_html_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    return "text/html" in content_type

def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "relevance": "Unknown",
            "opportunity_score": 0,
            "industry": "Unknown",
            "signals": [],
            "services": [],
            "risk_flags": [],
            "public_sector_hint": "Unknown",
            "agency_hints": [],
            "chunk_assessment": text[:500],
            "content_summary": text[:500],
            "public_sector_relevant": "Unknown",
            "relevant_agencies": [],
            "agency_pursuit_rationale": ""
        }

def dedupe_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for v in values:
        v = str(v).strip()
        if v and v not in seen:
            seen.add(v)
            result.append(v)
    return result

def get_with_retries(url: str, timeout: int = REQUEST_TIMEOUT) -> Optional[requests.Response]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES:
                print(f"Request failed after {MAX_RETRIES} attempts: {url} | {exc}")
                return None

            wait_time = RETRY_BASE_DELAY * attempt
            print(f"Retrying ({attempt}/{MAX_RETRIES}) after error: {url} | {exc}")
            time.sleep(wait_time)

    return None

def compute_content_hash(text: str) -> str:
    normalized = clean_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def utc_now_string() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# =========================
# Crawling functions
# =========================

def extract_internal_links(page_url: str, soup: BeautifulSoup, base_url: str) -> List[str]:
    links = []

    for a_tag in soup.find_all("a", href=True):
        raw_href = a_tag["href"].strip()
        full_url = urljoin(page_url, raw_href)
        full_url = normalize_url(full_url)

        if not full_url.startswith("http"):
            continue
        if not is_same_domain(base_url, full_url):
            continue
        if should_skip_url(full_url):
            continue

        links.append(full_url)

    return dedupe_preserve_order(links)

def crawl_site(
    start_url: str,
    max_pages: int = MAX_PAGES_DEFAULT,
    restrict_to_path: bool = False,
    progress_callback=None
) -> List[str]:
    start_url = normalize_url(start_url)
    visited: Set[str] = set()
    discovered: Set[str] = {start_url}
    queue = deque([start_url])
    results: List[str] = []

    base_domain = urlparse(start_url).netloc
    start_path = urlparse(start_url).path.rstrip("/")

    while queue and len(results) < max_pages:
        current_url = queue.popleft()

        if current_url in visited:
            continue

        if progress_callback:
            progress_callback(f"Crawling: {current_url}", len(results), max_pages)

        visited.add(current_url)

        response = get_with_retries(current_url)
        if not response:
            continue

        if not is_html_response(response):
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        results.append(current_url)

        internal_links = extract_internal_links(current_url, soup, start_url)

        for link in internal_links:
            if link in visited or link in discovered:
                continue

            parsed = urlparse(link)
            if parsed.netloc != base_domain:
                continue

            if restrict_to_path and start_path:
                if not parsed.path.startswith(start_path):
                    continue

            discovered.add(link)
            queue.append(link)

        time.sleep(CRAWL_DELAY_SECONDS)

    return results

# =========================
# Scraping functions
# =========================

def scrape_page(url: str) -> Optional[Dict[str, Any]]:
    response = get_with_retries(url)
    if not response:
        return None

    if not is_html_response(response):
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    title = soup.title.string.strip() if soup.title and soup.title.string else "No title"

    headings = []
    for tag_name in ["h1", "h2", "h3"]:
        for tag in soup.find_all(tag_name):
            text = clean_text(tag.get_text(" ", strip=True))
            if text:
                headings.append(f"{tag_name.upper()}: {text}")

    paragraphs = []
    for p_tag in soup.find_all("p"):
        text = clean_text(p_tag.get_text(" ", strip=True))
        if len(text) > 40:
            paragraphs.append(text)

    combined_text = "\n".join(headings + paragraphs)
    content_hash = compute_content_hash(combined_text)

    return {
        "url": url,
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs,
        "combined_text": combined_text,
        "content_hash": content_hash,
    }

def is_probably_useful_page(page: Dict[str, Any]) -> bool:
    text = page.get("combined_text", "").lower()
    title = page.get("title", "").lower()
    url = page.get("url", "").lower()

    for pattern in LOW_VALUE_PATTERNS:
        if pattern in title or pattern in url or pattern in text[:500]:
            return False

    return len(text) >= 200

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS
) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length and len(chunks) < MAX_CHUNKS_PER_PAGE:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]

        if end < text_length:
            last_period = chunk.rfind(". ")
            last_newline = chunk.rfind("\n")
            boundary = max(last_period, last_newline)
            if boundary > int(chunk_size * 0.6):
                chunk = chunk[:boundary + 1]
                end = start + len(chunk)

        chunks.append(chunk.strip())

        if end >= text_length:
            break

        start = max(end - overlap, 0)

    return chunks

# =========================
# DeepSeek analysis
# =========================

def call_deepseek_json(
    client: OpenAI,
    prompt: str,
    max_tokens: int = 1300
) -> Dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a B2B technology strategy analyst for Cognizant. "
                            "Return valid json only. No markdown. No code fences."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                stream=False,
            )

            text = response.choices[0].message.content or ""
            return safe_json_loads(text)

        except Exception as exc:
            if attempt == MAX_RETRIES:
                print(f"DeepSeek request failed after {MAX_RETRIES} attempts: {exc}")
                return {
                    "relevance": "Unknown",
                    "opportunity_score": 0,
                    "industry": "Unknown",
                    "signals": [],
                    "services": [],
                    "risk_flags": [f"API error: {str(exc)[:120]}"],
                    "public_sector_hint": "Unknown",
                    "agency_hints": [],
                    "chunk_assessment": "DeepSeek request failed.",
                    "content_summary": "",
                    "public_sector_relevant": "Unknown",
                    "relevant_agencies": [],
                    "agency_pursuit_rationale": ""
                }

            wait_time = RETRY_BASE_DELAY * attempt
            print(f"Retrying DeepSeek call ({attempt}/{MAX_RETRIES}) after error: {exc}")
            time.sleep(wait_time)

    return {
        "relevance": "Unknown",
        "opportunity_score": 0,
        "industry": "Unknown",
        "signals": [],
        "services": [],
        "risk_flags": ["Unexpected API failure"],
        "public_sector_hint": "Unknown",
        "agency_hints": [],
        "chunk_assessment": "DeepSeek request failed.",
        "content_summary": "",
        "public_sector_relevant": "Unknown",
        "relevant_agencies": [],
        "agency_pursuit_rationale": ""
    }

def analyze_chunk_with_deepseek(
    client: OpenAI,
    page_title: str,
    page_url: str,
    chunk_text_value: str,
    chunk_index: int,
    total_chunks: int
) -> Dict[str, Any]:
    prompt = f"""
Assess the following web content chunk for relevance to IT outsourcing and business opportunity for Cognizant.

Context:
- Page title: {page_title}
- Page URL: {page_url}
- Chunk number: {chunk_index} of {total_chunks}

Instructions:
1. Determine whether the content indicates a business opportunity related to IT outsourcing, managed services, consulting, systems integration, cloud modernization, data/AI, cybersecurity, application modernization, platform engineering, business process transformation, or public sector digital transformation.
2. Focus on explicit and implicit signals, not just keywords.
3. Ignore generic marketing fluff unless it indicates a real transformation need, spending trigger, policy change, operational pressure, modernization initiative, new digital program, regulatory burden, cost pressure, service gap, capability gap, or outsourcing need.
4. Identify whether the chunk appears relevant to government or public sector agencies.
5. Be concrete and commercially useful.
6. Return json only.

Return JSON with exactly these keys:
{{
  "relevance": "High|Medium|Low|None",
  "opportunity_score": 0,
  "industry": "string",
  "signals": ["string", "..."],
  "services": ["string", "..."],
  "risk_flags": ["string", "..."],
  "public_sector_hint": "Yes|No|Maybe",
  "agency_hints": ["string", "..."],
  "chunk_assessment": "max 90 words"
}}

Content chunk:
\"\"\"
{chunk_text_value}
\"\"\"
""".strip()

    return call_deepseek_json(client, prompt, max_tokens=700)

def synthesize_page_assessment(
    client: OpenAI,
    page_title: str,
    page_url: str,
    chunk_results: List[Dict[str, Any]],
    full_text_preview: str
) -> Dict[str, Any]:
    chunk_json = json.dumps(chunk_results, ensure_ascii=False, indent=2)

    prompt = f"""
You are preparing a concise go-to-market note for Cognizant.

Using the chunk-level analysis below, produce a final page-level assessment.

Return json with exactly these keys:
{{
  "overall_relevance": "High|Medium|Low|None",
  "overall_opportunity_score": 0,
  "industry": "string",
  "content_summary": "summary of the page content in no more than 150 words",
  "recommended_services": ["string", "..."],
  "public_sector_relevant": "Yes|No|Maybe",
  "relevant_agencies": ["string", "..."],
  "agency_pursuit_rationale": "brief explanation of why the listed public sector agencies are relevant targets",
  "implications": "brief note under 200 words covering the opportunity and implications for IT outsourcing",
  "pursuit_recommendation": "Strong pursue|Selective pursue|Monitor|No immediate action"
}}

Page title: {page_title}
Page URL: {page_url}

Chunk analyses:
{chunk_json}

Short text preview:
\"\"\"
{full_text_preview[:3000]}
\"\"\"

Rules:
- content_summary must be no more than 150 words.
- The summary should describe what the page is about, not sales advice.
- public_sector_relevant should be:
  - Yes: clearly relevant to one or more public sector agencies
  - Maybe: potentially relevant to public sector but not explicit
  - No: not meaningfully relevant to public sector pursuit
- relevant_agencies should list the public sector agencies or government bodies that would be logical targets for this opportunity.
- Only include agencies if there is a credible connection based on the page content.
- If there are no clear agencies, return an empty list.
- agency_pursuit_rationale should briefly explain why those agencies are relevant.
- If no agencies are relevant, return an empty string.
- Keep the implications field under 200 words.
- Be specific about why this matters for IT outsourcing.
- Mention likely demand triggers such as modernization, cloud, legacy renewal, compliance, cost pressure, AI adoption, operating model change, citizen/customer experience, cybersecurity, or managed services.
- Do not overstate certainty.
- If opportunity is weak, say so clearly.
- Return json only.
""".strip()

    return call_deepseek_json(client, prompt, max_tokens=1300)

def analyze_page_with_deepseek(client: OpenAI, page: Dict[str, Any], progress_callback=None) -> Dict[str, Any]:
    combined_text = page["combined_text"]
    chunks = chunk_text(combined_text)

    if not chunks:
        return {
            "URL": page["url"],
            "Title": page["title"],
            "Industry": "Unknown",
            "Overall Relevance": "None",
            "Opportunity Score": 0,
            "Content Summary": "",
            "Signals": "",
            "Recommended Services": "",
            "Risk Flags": "",
            "Public Sector Relevant": "No",
            "Relevant Agencies": "",
            "Agency Pursuit Rationale": "",
            "Pursuit Recommendation": "No immediate action",
            "Implications": "No meaningful textual content found on the page.",
            "Content Hash": page.get("content_hash", ""),
            "Analysis Timestamp": utc_now_string(),
            "Chunk Count": 0,
        }

    chunk_results = []
    for idx, chunk in enumerate(chunks, start=1):
        if progress_callback:
            progress_callback(f"Analyzing chunk {idx}/{len(chunks)} for: {page['title'][:50]}...")
        
        result = analyze_chunk_with_deepseek(
            client=client,
            page_title=page["title"],
            page_url=page["url"],
            chunk_text_value=chunk,
            chunk_index=idx,
            total_chunks=len(chunks),
        )
        chunk_results.append(result)
        time.sleep(0.5)

    final_assessment = synthesize_page_assessment(
        client=client,
        page_title=page["title"],
        page_url=page["url"],
        chunk_results=chunk_results,
        full_text_preview=combined_text[:8000],
    )

    all_signals = []
    all_services = []
    all_risk_flags = []
    all_agencies = []

    for chunk in chunk_results:
        all_signals.extend(chunk.get("signals", []))
        all_services.extend(chunk.get("services", []))
        all_risk_flags.extend(chunk.get("risk_flags", []))
        all_agencies.extend(chunk.get("agency_hints", []))

    all_services.extend(final_assessment.get("recommended_services", []))
    all_agencies.extend(final_assessment.get("relevant_agencies", []))

    all_signals = dedupe_preserve_order([str(x) for x in all_signals])
    all_services = dedupe_preserve_order([str(x) for x in all_services])
    all_risk_flags = dedupe_preserve_order([str(x) for x in all_risk_flags])
    all_agencies = dedupe_preserve_order([str(x) for x in all_agencies])

    return {
        "URL": page["url"],
        "Title": page["title"],
        "Industry": final_assessment.get("industry", "Unknown"),
        "Overall Relevance": final_assessment.get("overall_relevance", "Unknown"),
        "Opportunity Score": final_assessment.get("overall_opportunity_score", 0),
        "Content Summary": final_assessment.get("content_summary", ""),
        "Signals": " | ".join(all_signals),
        "Recommended Services": " | ".join(all_services),
        "Risk Flags": " | ".join(all_risk_flags),
        "Public Sector Relevant": final_assessment.get("public_sector_relevant", "Unknown"),
        "Relevant Agencies": " | ".join(all_agencies),
        "Agency Pursuit Rationale": final_assessment.get("agency_pursuit_rationale", ""),
        "Pursuit Recommendation": final_assessment.get("pursuit_recommendation", "Monitor"),
        "Implications": final_assessment.get("implications", ""),
        "Content Hash": page.get("content_hash", ""),
        "Analysis Timestamp": utc_now_string(),
        "Chunk Count": len(chunks),
    }

# =========================
# Flask Routes
# =========================

@app.route('/')
def index():
    """Render the main input form."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle the analysis request."""
    url = request.form.get('url', '').strip()
    max_pages = int(request.form.get('max_pages', MAX_PAGES_DEFAULT))
    restrict_to_path = request.form.get('restrict_to_path') == 'on'
    
    if not url:
        flash('Please enter a URL to analyze', 'error')
        return redirect(url_for('index'))
    
    # Store parameters in session for the progress page
    session['analysis_params'] = {
        'url': url,
        'max_pages': max_pages,
        'restrict_to_path': restrict_to_path
    }
    session['analysis_results'] = None
    session['analysis_status'] = 'starting'
    session['analysis_logs'] = []
    
    return redirect(url_for('progress'))

@app.route('/progress')
def progress():
    """Show progress page with live updates."""
    if 'analysis_params' not in session:
        return redirect(url_for('index'))
    
    return render_template('progress.html')

@app.route('/api/run-analysis')
def run_analysis():
    """API endpoint to run the analysis and stream progress."""
    from flask import Response, stream_with_context
    
    params = session.get('analysis_params', {})
    if not params:
        return jsonify({'error': 'No analysis parameters found'}), 400
    
    def generate():
        def log_message(msg, progress=None):
            nonlocal log_lines
            log_lines.append(msg)
            session['analysis_logs'] = log_lines[-50:]  # Keep last 50 lines
            session.modified = True
            
            data = {'log': msg}
            if progress is not None:
                data['progress'] = progress
            yield f"data: {json.dumps(data)}\n\n"
        
        log_lines = []
        
        try:
            # Initialize DeepSeek client
            log_message("Initializing DeepSeek client...")
            client = get_deepseek_client()
            
            # Crawl the site
            log_message(f"Starting crawl of {params['url']}...")
            urls = crawl_site(
                start_url=params['url'],
                max_pages=params['max_pages'],
                restrict_to_path=params['restrict_to_path'],
                progress_callback=lambda msg, current, total: log_message(msg, int((current/total)*30))
            )
            
            log_message(f"Found {len(urls)} pages to analyze", 30)
            
            # Scrape pages
            scraped_pages = []
            for i, url in enumerate(urls):
                log_message(f"Scraping: {url}", 30 + int((i/len(urls))*20))
                
                page_data = scrape_page(url)
                if page_data and is_probably_useful_page(page_data):
                    scraped_pages.append(page_data)
            
            log_message(f"Found {len(scraped_pages)} useful pages to analyze", 50)
            
            if not scraped_pages:
                log_message("No useful pages found to analyze", 100)
                session['analysis_results'] = []
                session['analysis_status'] = 'completed'
                yield f"data: {json.dumps({'complete': True, 'redirect': '/results'})}\n\n"
                return
            
            # Analyze pages
            results = []
            for i, page in enumerate(scraped_pages):
                log_message(f"Analyzing page {i+1}/{len(scraped_pages)}: {page['title'][:50]}...", 
                           50 + int((i/len(scraped_pages))*45))
                
                result = analyze_page_with_deepseek(
                    client, 
                    page,
                    lambda msg: log_message(f"  {msg}")
                )
                results.append(result)
            
            log_message("Analysis complete!", 95)
            
            # Convert to DataFrame for sorting/filtering
            df = pd.DataFrame(results)
            
            # Sort by opportunity score
            relevance_rank = {"High": 4, "Medium": 3, "Low": 2, "None": 1, "Unknown": 0}
            df["Relevance Rank"] = df["Overall Relevance"].map(relevance_rank).fillna(0)
            df = df.sort_values(
                by=["Opportunity Score", "Relevance Rank", "Title"],
                ascending=[False, False, True],
            ).drop(columns=["Relevance Rank"])
            
            # Store results in session
            session['analysis_results'] = df.to_dict('records')
            session['analysis_status'] = 'completed'
            session.modified = True
            
            log_message("Redirecting to results...", 100)
            yield f"data: {json.dumps({'complete': True, 'redirect': '/results'})}\n\n"
            
        except Exception as e:
            log_message(f"ERROR: {str(e)}")
            session['analysis_status'] = 'error'
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/results')
def results():
    """Display the analysis results."""
    results_data = session.get('analysis_results', [])
    
    if not results_data:
        flash('No results available. Please run an analysis first.', 'warning')
        return redirect(url_for('index'))
    
    # Calculate summary statistics
    df = pd.DataFrame(results_data)
    
    summary = {
        'total_pages': len(df),
        'high_opportunity': len(df[df['Opportunity Score'] >= 70]),
        'medium_opportunity': len(df[(df['Opportunity Score'] >= 40) & (df['Opportunity Score'] < 70)]),
        'low_opportunity': len(df[df['Opportunity Score'] < 40]),
        'public_sector_count': len(df[df['Public Sector Relevant'] == 'Yes']),
        'avg_score': round(df['Opportunity Score'].mean(), 1) if not df.empty else 0,
        'industries': df['Industry'].value_counts().head(5).to_dict(),
        'top_pages': df.nlargest(5, 'Opportunity Score')[['Title', 'Opportunity Score', 'Overall Relevance', 'Public Sector Relevant']].to_dict('records')
    }
    
    return render_template('results.html', 
                         results=results_data, 
                         summary=summary,
                         results_json=json.dumps(results_data))

@app.route('/api/results')
def api_results():
    """API endpoint to get results as JSON."""
    results_data = session.get('analysis_results', [])
    return jsonify(results_data)

@app.route('/export')
def export_results():
    """Export results as JSON or CSV."""
    format_type = request.args.get('format', 'json')
    results_data = session.get('analysis_results', [])
    
    if format_type == 'json':
        return jsonify(results_data)
    elif format_type == 'csv':
        df = pd.DataFrame(results_data)
        return Response(
            df.to_csv(index=False),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=analysis_results.csv'}
        )
    
    return jsonify({'error': 'Invalid format'}), 400

@app.route('/clear')
def clear_session():
    """Clear the session and start over."""
    session.clear()
    flash('Session cleared. Start a new analysis.', 'info')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', 5000)))
