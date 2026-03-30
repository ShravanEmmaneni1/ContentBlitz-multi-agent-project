# ContentBlitz-multi-agent-project
Multi-agent pipeline that automates research (SerpAPI + OpenAI summarization), SEO blog drafts (GPT), LinkedIn posts (GPT), and images (DALL·E), with a local FAISS-style vector index (OpenAI embeddings; uses faiss-cpu when available, otherwise equivalent NumPy cosine search).

Setup
cd contentblitz
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: `OPENAI_API_KEY` (required) and `SERPAPI_API_KEY` (required when `SERPAPI_ENABLED=1`).
Run
streamlit run app.py
Agents
Agent	Role	Backend
Research	Grounded summaries and angles	SerpAPI + OpenAI summarization
Blog writing	Title, meta, markdown, keywords	GPT
LinkedIn	Post, hooks, hashtags	GPT
Image creation	Hero / social visuals	DALL·E 3
Optional env: OPENAI_MODEL, SERPAPI_ENGINE, SERPAPI_ENABLED, MCP_ENABLED, VECTOR_STORE_PATH (aliases: CHROMA_PATH, FAISS_PATH), OPENAI_EMBEDDING_MODEL.

Layout
app.py — Streamlit UI
orchestrator.py — pipeline wiring
mcp_agent_server.py — MCP (FastMCP) tool-based orchestration server
vector_store.py — local FAISS/NumPy store (./data/faiss by default)
ingest_se401k.py — chunk/embed se401kbrochure.pdf (run python ingest_se401k.py; pipeline also auto-ingests if the PDF is present)
rag.py — retrieval query helper for self-employed / SEP / solo 401(k) context
agents/ — one module per agent
Place se401kbrochure.pdf in the project root (or set SE401K_BROCHURE_PATH). Retrieval favors brochure chunks for blog, LinkedIn, and image prompts.
