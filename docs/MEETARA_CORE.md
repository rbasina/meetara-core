# me²TARA Core

## The Vision: AI That Respects You

In a world where artificial intelligence increasingly demands your data, your privacy, and your trust in distant servers, **me²TARA** stands apart. We believe that intelligent assistance shouldn't come at the cost of your privacy, your autonomy, or your peace of mind.

me²TARA is an **offline-first, privacy-focused, and empathetic AI assistant** — built from the ground up to run entirely on your own hardware, understand your emotional context, and provide expert-level assistance across over 100 knowledge domains.

---

## What Does "me²TARA" Mean?

The name **me²TARA** carries deep significance:

- **me²** (me-squared) — This AI is an extension of *you*. It amplifies your capabilities, learns your context, and serves your needs — not a corporation's interests. The "squared" represents exponential empowerment.

- **TARA** — In Sanskrit, "Tara" (तारा) means "star" or "one who guides across." Like a guiding star, me²TARA illuminates your path through complex information, helping you navigate healthcare decisions, legal questions, educational challenges, and life's countless domains of knowledge.

Together, **me²TARA** represents *your personal guiding star* — an AI that belongs to you, works for you, and stays with you.

---

## Core Philosophy

### 🔒 Privacy by Design, Not by Promise

Most AI assistants process your queries on remote servers. Your questions about health symptoms, financial troubles, relationship struggles, and private thoughts traverse the internet, stored in databases you'll never see, analyzed in ways you'll never know.

**me²TARA takes a fundamentally different approach.**

Every computation happens on your machine. Your documents stay in your `vectorstore/` folder. Your conversations remain in your local session. Your fine-tuned models run from your `.cache/huggingface/` directory. There is no cloud. There is no "trust us" — there is only verifiable, auditable, local processing.

When you ask me²TARA about a sensitive medical condition, that question never leaves your computer. When you upload confidential business documents, they're chunked, embedded, and stored in your local ChromaDB instance. When you discuss personal struggles, those words exist only in your memory, not ours.

**Privacy isn't a feature we added. It's the foundation we built upon.**

---

### 🌐 Offline-First: Intelligence Without Internet

The internet is not always available. The internet is not always trustworthy. The internet is not always fast.

me²TARA is designed to function completely offline:

- **Local Language Models**: Our custom-trained Qwen3 models (1.7B, 4B-instruct, 4B-thinking, 8B parameters) run entirely on your hardware using efficient GGUF quantization
- **Local Vector Database**: ChromaDB stores your knowledge base with Snappy compression and SQLite optimizations
- **Local Embeddings**: Document embeddings are generated on-device
- **Local Processing**: No API calls, no tokens, no rate limits, no subscriptions

Whether you're on an airplane, in a remote location, or simply prefer to disconnect — me²TARA remains fully functional. Your AI assistant doesn't abandon you when WiFi does.

---

### 💚 Empathetic Intelligence: AI That Understands Context

Information without empathy is just data. me²TARA is built to understand not just *what* you're asking, but *how* you're feeling when you ask it.

**Emotion-Aware Responses**

me²TARA integrates emotion detection to adapt its communication style:

- When you're **stressed**, responses become calmer, more reassuring, and more structured
- When you're **confused**, explanations become clearer, with more examples and simpler language
- When you're **curious**, responses become richer, with deeper exploration and related concepts
- When you're **frustrated**, responses become more direct, acknowledging the difficulty and focusing on solutions

This isn't about manipulation — it's about communication. A good teacher adjusts their explanation based on the student's state. A good doctor considers the patient's emotional readiness. A good assistant recognizes that the *same information* delivered differently can be the difference between helping and overwhelming.

**Multi-Language Understanding**

Empathy transcends language barriers. me²TARA supports 15+ languages, allowing you to communicate in the language most comfortable to you — because cognitive load matters, and struggling with a second language while dealing with complex topics creates unnecessary friction.

---

## Domain Expertise: Knowledge That Matters

me²TARA isn't a general-purpose chatbot trying to be everything to everyone. It's a **domain-aware knowledge system** with specialized understanding across critical areas of life:

### Safety-Critical Domains (Highest Accuracy)
- **Healthcare**: General health, mental health, nutrition, chronic conditions, medication management
- **Legal & Financial**: Legal assistance, insurance, real estate, financial planning
- **Emergency**: Crisis management, disaster preparedness, emergency response

### Expert Domains (Professional-Grade)
- **Business**: Entrepreneurship, marketing, sales, project management, strategy
- **Education**: Academic tutoring, skill development, exam preparation, research assistance
- **Technology**: Programming, AI/ML, cybersecurity, software development

### Quality Domains (Comprehensive Coverage)
- **Personal Life**: Parenting, relationships, time management, work-life balance
- **Creative**: Writing, storytelling, photography, music, art appreciation
- **Wellness**: Psychology, yoga, life coaching, fitness, stress management

Each domain has:
- **Curated keywords** for intelligent routing
- **Specialized prompting** for domain-appropriate responses
- **Tiered validation** ensuring safety-critical domains receive extra scrutiny
- **Dedicated vector stores** for domain-specific document retrieval

---

## The RAG Advantage: Your Knowledge, Amplified

**Retrieval-Augmented Generation (RAG)** is the technical foundation that makes me²TARA truly yours.

### How It Works

1. **You upload documents** — PDFs, research papers, company policies, medical records, study materials
2. **me²TARA processes them** — Intelligent chunking preserves context, embeddings capture meaning
3. **Your knowledge base grows** — Each domain maintains its own searchable vector store
4. **Queries retrieve relevant context** — When you ask a question, me²TARA finds the most relevant chunks from YOUR documents
5. **The LLM generates grounded answers** — Responses are based on your actual documents, not hallucinated from training data

### Why This Matters

Traditional LLMs have a knowledge cutoff date. They hallucinate. They can't access your specific documents, your company's policies, your medical history, your course materials.

me²TARA's RAG system means:
- **Your textbooks become queryable** — "What does Chapter 7 say about thermodynamics?"
- **Your company policies become accessible** — "What's our remote work policy?"
- **Your medical records become understandable** — "Explain my latest blood test results"
- **Your research becomes searchable** — "Find all papers mentioning neural plasticity"

The AI doesn't just generate text — it retrieves YOUR knowledge and augments it with language understanding.

---

## Technical Excellence

### Model Selection: Right Tool for the Job

me²TARA offers multiple model configurations, each optimized for different use cases:

| Model | Parameters | Best For |
|-------|------------|----------|
| **meetara-qwen3-1.7b** | 1.7B | Fast responses, simple queries, low-resource devices |
| **meetara-qwen3-4b-instruct** | 4B | Detailed explanations, instruction-following |
| **meetara-qwen3-4b-thinking** | 4B | Complex reasoning, multi-step problems |
| **meetara-qwen3-8b** | 8B | Advanced understanding, nuanced responses |

All models are:
- **Custom fine-tuned** on domain-specific data
- **Quantized to GGUF** for efficient local execution
- **Hosted on Hugging Face** for easy download and updates
- **Automatically cached** for offline availability

### Architecture Principles

- **Modular Design**: Each component (RAG, LLM, emotion detection, domain routing) is independent and replaceable
- **Config-Driven**: YAML files control domains, keywords, tiers, and models — no code changes needed for customization
- **Lazy Loading**: Models load only when needed, preserving system resources
- **Graceful Degradation**: If one component fails, others continue functioning

---

## The me²TARA Promise

We believe AI should:

1. **Respect your privacy** — Your data is yours. Period.
2. **Work offline** — Intelligence shouldn't require internet.
3. **Understand context** — Emotional awareness makes better assistance.
4. **Provide expertise** — Domain-specific knowledge, not generic responses.
5. **Empower you** — Your knowledge, amplified by AI.

me²TARA isn't just another AI assistant. It's a philosophy made software — the belief that the most powerful AI is the one that serves you completely, privately, and empathetically.

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/meetara-lab/meetara-core.git

# Install dependencies
pip install -r requirements.txt

# Start the backend
python main.py

# Start the frontend (in another terminal)
cd meetara-ui && npm install && npm run dev
```

Your personal AI assistant awaits at `http://localhost:3000`.

Upload your documents. Ask your questions. Keep your privacy.

**Welcome to me²TARA — your guiding star.**

---

*"The best AI is not the one that knows everything, but the one that helps you understand what matters to you."*

— The me²TARA Team

