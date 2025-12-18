# Agentic RAG for Civil Law Systems

This repository contains the code and benchmark scripts for the paper:

**"Agentic RAG for Civil Law Systems: Tool-Use Dynamics in Korean Bar Exam Question Answering"**

*Eul Young Lim and Jihun Park*
Chungnam National University, South Korea

## Abstract

Legal problem-solving in civil law systems requires hierarchical, multi-step exploration from codified statutes to interpretive case law. This study applies an agentic RAG architecture to the civil law domain, where LLMs autonomously select and iteratively invoke tools, and analyzes how tool-use dynamics affect performance. Using the Model Context Protocol (MCP), our system provides five tools—statute search, article lookup, case search, judgment lookup, and statute TOC browsing—emulating legal professionals' research workflows. Evaluated on 150 multiple-choice questions from the 2025 Korean Bar Examination, agentic RAG achieved **96.7% accuracy with GPT-5.1 (High)** and **94.7% with Claude Sonnet 4.5 (Max-Thinking)**, representing significant 10.7 percentage point improvements over Naïve RAG (McNemar's test, *p* < .01). However, Gemini 2.5 Pro showed a 16.0 percentage point degradation (*p* < .001). Analysis revealed three distinct tool-use patterns: GPT-5.1 demonstrated *Deep Exploration* (63% judgment lookup ratio); Claude showed *Efficient Utilization* (42% lookup ratio with comparable performance); Gemini exhibited *Search-Lookup Disconnection* (0% lookup ratio) and *Tool Aversion*. These tool-use failures caused performance degradation. Our findings demonstrate that agentic RAG effectiveness depends critically on model-specific tool-use propensities, highlighting the importance of evaluating tool-use dynamics beyond accuracy when building legal AI systems.

## Repository Structure

```
legal_mcp_repo/
├── legal_mcp/                    # MCP Server implementation
│   └── src/
│       ├── server.py             # Main FastMCP server
│       ├── config.py             # Configuration management
│       ├── elasticsearch/        # Elasticsearch client and queries
│       │   ├── client.py
│       │   └── queries.py
│       ├── tools/                # MCP tool implementations
│       │   ├── search_cases.py
│       │   ├── search_statutes.py
│       │   ├── get_case_content.py
│       │   ├── get_statute_content.py
│       │   └── list_statute_articles.py
│       └── utils/                # Utility functions
│           ├── embedding.py
│           ├── formatters.py
│           └── rrf_fusion.py
├── benchmark/                    # Benchmark scripts and results
│   ├── legal_close_book/         # Closed Book experiments
│   │   ├── benchmark.csv         # 150 Korean Bar Exam questions
│   │   ├── closed_book_benchmark_*.py
│   │   └── result/
│   ├── legal_naive_rag/          # Naïve RAG experiments
│   │   ├── rag_benchmark_*.py
│   │   └── result/
│   └── legal_mcp_rag/            # Agentic RAG experiments
│       ├── mcp_benchmark_*.py
│       └── result/
├── docs/                         # Documentation
│   ├── tools-spec.md             # MCP tool specifications
│   ├── config-spec.md            # Configuration documentation
│   └── elasticsearch-fields.md   # Index field documentation
├── requirements.txt
├── .env.example
└── README.md
```

## Dataset

The benchmark dataset and legal database are available on Hugging Face:

**[ducut91/legal_mcp](https://huggingface.co/datasets/ducut91/legal_mcp)**

The dataset includes:
- **Benchmark Questions**: 150 multiple-choice questions from the 14th Korean Bar Examination (2025)
  - Civil Law (민사법): 70 questions
  - Criminal Law (형사법): 40 questions
  - Public Law (공법): 40 questions
- **Court Cases**: 193,276 Korean court judgments (including 29,730 Constitutional Court decisions)
- **Statutes**: 5,474 current Korean statutes with 200,633 individual articles

## MCP Server

### Public MCP Server

A public MCP server is available for testing:

```
https://mcp.crow-tit.com/sse
```

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/LimEulYoung/legal_mcp.git
cd legal_mcp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
```
# Elasticsearch Configuration
ES_HOST=your_elasticsearch_host
ES_PORT=9200
ES_SCHEME=http
ES_USER=your_username
ES_PASSWORD=your_password

# Upstage API Configuration (REQUIRED for embeddings)
UPSTAGE_API_KEY=your_upstage_api_key

# Benchmark API Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
```

### MCP Tools

The server provides five tools for legal research:

| Tool | Description |
|------|-------------|
| `search_cases` | Search Korean court cases by keywords, with filters for court, date, and statute references |
| `get_case_content` | Retrieve full judgment text by case number |
| `search_statutes` | Search Korean statutes by name or legal concept |
| `get_statute_content` | Retrieve statute articles (full or specific articles) |
| `list_statute_articles` | List table of contents for a statute |

#### Quick Access Statute IDs

| Statute | ID | Statute | ID |
|---------|----|---------|----|
| Constitution (헌법) | 1444 | Framework Act on Administration (행정기본법) | 14041 |
| Civil Act (민법) | 1706 | Administrative Procedure Act (행정절차법) | 1362 |
| Commercial Act (상법) | 1702 | Administrative Litigation Act (행정소송법) | 1363 |
| Civil Procedure Act (민사소송법) | 1700 | Constitutional Court Act (헌법재판소법) | 11233 |
| Criminal Act (형법) | 1692 | | |
| Criminal Procedure Act (형사소송법) | 1671 | | |

## Running Benchmarks

### Prerequisites

1. Ensure you have the required API keys configured in `.env`
2. Download the benchmark dataset from Hugging Face or use the included `benchmark.csv`

### Command-Line Arguments

All benchmark scripts support the following arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--limit` | Number of questions to run | All (150) |
| `--workers` | Number of parallel workers | 3 |

### Closed Book Benchmark

```bash
cd benchmark/legal_close_book

# Run all 150 questions
python closed_book_benchmark_gpt_5.1_high.py

# Run first 10 questions with 5 parallel workers
python closed_book_benchmark_gpt_5.1_high.py --limit 10 --workers 5

# Other models
python closed_book_benchmark_claude_sonnet_4.5_max_thinking.py
python closed_book_benchmark_gemini_2.5_high.py
```

### Naïve RAG Benchmark

```bash
cd benchmark/legal_naive_rag

# Run all 150 questions
python rag_benchmark_gpt_5.1_high.py

# Run first 10 questions with 5 parallel workers
python rag_benchmark_gpt_5.1_high.py --limit 10 --workers 5

# Other models
python rag_benchmark_claude_4.5_sonnet_max_thinking.py
python rag_benchmark_gemini_2.5_high.py
```

### Agentic RAG Benchmark (MCP)

```bash
cd benchmark/legal_mcp_rag

# Run all 150 questions
python mcp_benchmark_gpt_5.1_high.py

# Run first 10 questions with 5 parallel workers
python mcp_benchmark_gpt_5.1_high.py --limit 10 --workers 5

# Other models
python mcp_benchmark_claude_sonnet_4.5_max_thinking.py
python mcp_benchmark_gemini_2.5_pro_high.py

# Ablation study with explicit guidance
python mcp_benchmark_gemini_2.5_pro_high_guided.py
```

### Benchmark Results

Results are saved as JSON files in the `result/` subdirectory of each benchmark folder.

## Results Summary

### Overall Performance (Accuracy %)

| Model | Closed Book | Naïve RAG | Agentic RAG | Δ Naïve |
|-------|-------------|-----------|-------------|---------|
| Claude 4.5 (Max-Think) | 51.33 | 84.00 | **94.67** | +10.67 |
| GPT-5.1 (High) | 54.00 | 86.00 | **96.67** | +10.67 |
| Gemini 2.5 (High) | 60.67 | **89.33** | 73.33 | -16.00 |

### Tool-Use Dynamics

| Model | Avg Calls | search_cases | get_case_content | Lookup Ratio |
|-------|-----------|--------------|------------------|--------------|
| Claude 4.5 (Max-Think) | 10.29 | 837 | 355 | 42.4% |
| GPT-5.1 (High) | 16.61 | 1,115 | 706 | **63.3%** |
| Gemini 2.5 (High) | 3.29 | 382 | 0 | 0.0% |

**Key Findings:**
- **Deep Exploration** (GPT-5.1): High judgment lookup ratio (63.3%) correlates with best performance
- **Efficient Utilization** (Claude): Achieves comparable accuracy with fewer lookups
- **Search-Lookup Disconnection** (Gemini): Zero lookup ratio despite searching, leading to performance degradation


## License

This project is licensed under the MIT License.

## Contact

- Eul Young Lim - ey_lim@o.cnu.ac.kr
- Jihun Park (Corresponding Author) - jihun.park@cnu.ac.kr

Chungnam National University, Daejeon, South Korea

