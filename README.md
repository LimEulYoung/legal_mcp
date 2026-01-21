# Agentic RAG for Civil Law Systems

This repository contains the code, benchmark scripts, and supplementary materials for reproducibility.

## Paper Supplements

The full appendix (excluded from the main paper due to page limits) is available here:

**[Download Full Appendix (PDF)](appendix.pdf)** — Tool specifications, prompt templates, detailed experimental results, and additional analyses.

---

## Overview

![Agentic RAG Architecture](assets/figure2.png)

*Comparison between a human lawyer's iterative research process (left) and the proposed agentic RAG system (right). Both follow a similar pattern: searching for relevant statutes and cases, then looking up full content.*

## Repository Structure

```
├── README.md
├── appendix.pdf              # Full paper appendix
├── requirements.txt
├── .env.example
│
├── src/                      # MCP Server implementation
│   ├── server.py
│   ├── config.py
│   ├── elasticsearch/
│   ├── tools/
│   └── utils/
│
├── benchmark/
│   ├── data/                 # Benchmark dataset
│   │   └── benchmark.csv     # 150 Korean Bar Exam questions
│   ├── closed_book/          # Closed Book experiments
│   ├── naive_rag/            # Naïve RAG experiments
│   ├── agentic_rag/          # Agentic RAG experiments
│   └── results/              # Experiment results
│
├── docs/                     # Documentation
│
└── assets/                   # Figures and images
```

## Dataset

| Data | Description | Source |
|------|-------------|--------|
| Benchmark Questions | 150 MCQs from 14th Korean Bar Exam (2025): Civil (70), Criminal (40), Public (40) | [Ministry of Justice](https://www.corrections.go.kr/bbs/moj/150/591294/artclView.do) |
| Court Cases | 193,276 Korean court judgments (incl. 29,730 Constitutional Court decisions) | [Korea Open Law Information](https://open.law.go.kr) |
| Statutes | 5,474 current Korean statutes with 200,633 individual articles | [Korea Open Law Information](https://open.law.go.kr) |

## MCP Tools

The server provides five tools for legal research:

| Tool | Description |
|------|-------------|
| `search_cases` | Search Korean court cases by keywords, with filters for court, date, and statute references |
| `get_case_content` | Retrieve full judgment text by case number |
| `search_statutes` | Search Korean statutes by name or legal concept |
| `get_statute_content` | Retrieve statute articles (full or specific articles) |
| `list_statute_articles` | List table of contents for a statute |

### Quick Access Statute IDs

| Statute | ID | Statute | ID |
|---------|----|---------|----|
| Constitution | 1444 | Framework Act on Administration | 14041 |
| Civil Act | 1706 | Administrative Procedure Act | 1362 |
| Commercial Act | 1702 | Administrative Litigation Act | 1363 |
| Civil Procedure Act | 1700 | Constitutional Court Act | 11233 |
| Criminal Act | 1692 | | |
| Criminal Procedure Act | 1671 | | |

## Running Benchmarks

### Prerequisites

1. Configure API keys in `.env` (see `.env.example`)
2. Install dependencies: `pip install -r requirements.txt`

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--limit` | Number of questions to run | All (150) |
| `--workers` | Number of parallel workers | 3 |

### Examples

```bash
# Closed Book
python benchmark/closed_book/closed_book_benchmark_gpt_5.1_high.py --limit 10 --workers 5

# Naive RAG
python benchmark/naive_rag/rag_benchmark_gpt_5.1_high.py --limit 10 --workers 5

# Agentic RAG (Ours)
python benchmark/agentic_rag/mcp_benchmark_gpt_5.1_high.py --limit 10 --workers 5
```

## Results Summary

### Overall Performance (Accuracy %)

| Model | Closed Book | Naïve RAG | Agentic RAG |
|-------|-------------|-----------|-------------|
| Claude 4.5 (Max-Think) | 51.33 | 84.00 | **94.67** |
| GPT-5.1 (High) | 54.00 | 86.00 | **96.67** |
| Gemini 2.5 (High) | 60.67 | **89.33** | 73.33 |

### Key Findings

| Pattern | Model | Observation |
|---------|-------|-------------|
| **Deep Exploration** | GPT-5.1 | Highest lookup ratio (63.3%) → Best accuracy (96.67%) |
| **Efficient Utilization** | Claude | Comparable accuracy (94.67%) with fewer lookups (42.4%) |
| **Search-Lookup Disconnection** | Gemini | Zero lookups → Agentic RAG worse than Naïve RAG |

> See [appendix.pdf](appendix.pdf) for detailed results and analyses.

## License

This project is licensed under the MIT License.
