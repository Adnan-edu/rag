# 🚀 Project Overview: Building Knowledge Graphs with LightRAG

This project provides a comprehensive walkthrough of using the `LightRAG` library to automatically construct and query a Knowledge Graph (KG) from unstructured text. The notebook uses a document about Greek mythology (`Greek-gods-story.pdf`) as its source data. The entire process, from data ingestion to interactive graph visualization, is demonstrated, highlighting the power and simplicity of the `LightRAG` framework.

---

### 💡 Core Technology: LightRAG

`LightRAG` is presented as a lightweight, efficient Retrieval-Augmented Generation (RAG) framework. The key advantages highlighted are:

*   **⚡ Simplicity & Speed:** Designed for easy setup and fast performance, making it ideal for rapid prototyping and deployment.
*   **💸 Cost-Effective:** Its lightweight nature helps reduce computational overhead and associated costs.
*   **🧩 Flexible Integrations:** It can easily connect with various data sources and Large Language Models (LLMs).
*   **🤖 Key Capability:** The notebook's core focus is on its ability to perform **Automated Knowledge Graph extraction**, identifying entities and their relationships from raw text to build a structured, queryable knowledge base.

---

### ⚙️ Workflow & Implementation Details

The notebook follows a clear, step-by-step implementation plan.

#### 1. Environment Setup & Dependencies 🛠️

The initial setup involves preparing the Python environment.

*   **Dependencies:** Key libraries are installed using `pip`:
    *   `lightrag-hku==1.1.1`: The core framework.
    *   `PyPDF2`: For parsing and extracting text from PDF files.
    *   `networkx` & `pyvis`: Essential for graph manipulation and creating interactive visualizations.
    *   `python-dotenv` & `google-colab-userdata`: For securely managing API keys.
*   **API Key Management:** The OpenAI API key is fetched securely using `userdata.get()` and set as an environment variable, which is a security best practice.
    ```python
    # Set the OpenAI API key in the environment
    api_key = userdata.get('OPENAI_API_KEY')
    os.environ['OPENAI_API_KEY'] = api_key
    ```
*   **Asyncio Compatibility:** `nest_asyncio` is used to allow `asyncio` event loops to run within a Jupyter/Colab environment, which is necessary for `LightRAG`'s asynchronous operations.
    ```python
    import nest_asyncio
    nest_asyncio.apply()
    ```

#### 2. Data Preparation Pipeline 📄➡️📝

The workflow begins by converting the unstructured data source (a PDF) into a clean text format.

1.  **Load PDF:** The `Greek-gods-story.pdf` file is opened in binary-read mode (`"rb"`).
2.  **Extract Text:** `PyPDF2.PdfReader` iterates through each page of the PDF, and `page.extract_text()` pulls the raw text content.
3.  **Save as TXT:** The extracted text lines are written to a new file, `greek_gods.txt`. This provides a standardized, clean input for the `LightRAG` system.

#### 3. Automated Knowledge Graph Construction 🧠

This is the central and most impressive step, where `LightRAG` automatically builds the knowledge base from the text.

*   **Initialization:** A `LightRAG` object is instantiated, configured with a working directory and an LLM. The notebook uses `gpt_4o_complete`, indicating that **GPT-4o** is the engine for entity extraction and generation.
    ```python
    rag = LightRAG(
        working_dir="./working_dir",
        llm_model_func=gpt_4o_complete
    )
    ```
*   **Ingestion and Processing:** The entire knowledge graph construction is triggered by a single command:
    ```python
    with open("greek_gods.txt", "r") as file:
        rag.insert(file.read())
    ```
*   The detailed output logs (Page 4) reveal the multi-stage process happening under the hood:
    1.  **Chunking:** The input text is divided into smaller, manageable chunks.
    2.  **Entity & Relationship Extraction:** The LLM processes each chunk to identify entities (e.g., "Zeus", "Apollo") and the relationships between them (e.g., "son of", "stole from").
    3.  **Embedding Generation:** Vector embeddings are created for the text chunks to enable semantic search.
    4.  **Data Insertion:** The extracted entities, relationships, and embeddings are inserted into the `LightRAG` knowledge base, which is persisted as a `graphml` file.

#### 4. Retrieval-Augmented Generation (RAG) Strategies 🔍

The notebook demonstrates how to query the newly built KG using different retrieval strategies by changing the `mode` parameter in the `rag.query()` method. The query used for all examples is **"Who is Hermes"**.

*   **Naive RAG (`mode="naive"`):** This likely performs a standard semantic search on the text chunks. The response is a single block of text summarizing information about Hermes from the source document.

*   **Local Search RAG (`mode="local"`):** This mode provides a more structured, graph-aware retrieval. It first identifies high-level and low-level keywords related to the query and then generates a response. The keywords (`"God of trade"`, `"God of thieves"`) suggest it focuses on the immediate neighborhood of the "Hermes" entity within the KG.

*   **Global Search RAG (`mode="global"`):** This mode expands the retrieval scope. The keywords generated (`"Mythology"`, `"Olympian gods"`) are broader, indicating that the RAG system traverses more of the graph to gather wider context. The resulting answer is a more holistic overview of Hermes' role in the broader mythology.

*   **Hybrid Search RAG (`mode="hybrid"`):** This mode likely combines the precision of local search with the contextual breadth of global search. The keywords (`"Cultural significance"`, `"Caduceus"`) and the comprehensive answer suggest a balanced approach to deliver a highly relevant and detailed response.

#### 5. Knowledge Graph Visualization 🌐

A key feature of this workflow is the ability to visualize the generated knowledge graph, which is invaluable for understanding the data structure and debugging.

*   **Loading the Graph:** The KG, which `LightRAG` saved as `graph_chunk_entity_relation.graphml`, is loaded into a `networkx` graph object.
*   **Interactive Visualization:** The `pyvis` library is used to convert the `networkx` graph into a dynamic, interactive HTML file. The code customizes the graph by adding descriptions to nodes and edges, which appear on hover.
    ```python
    # Load the graphml file created by LightRAG
    G = nx.read_graphml(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    # Create an interactive network with pyvis
    net = Network(notebook=True, cdn_resources="remote")
    net.from_nx(G)
    net.show("graph.html")
    ```
*   **Subgraph Exploration:** The notebook demonstrates a powerful analysis technique by isolating and visualizing a subgraph. It targets the **"ZEUS"** node, finds all its direct neighbors, and creates a new visualization showing only Zeus and its immediate relationships. This is excellent for focused exploration of specific entities.

---

### 🎯 Key Findings & Takeaways

*   **End-to-End Automation:** `LightRAG` successfully automates the complex pipeline of KG creation from unstructured text with minimal code.
*   **Intelligent Retrieval:** The different query modes (`local`, `global`, `hybrid`) provide powerful control over the retrieval scope, allowing users to tailor the RAG output from a specific, focused answer to a broad, contextual overview.
*   **Transparency and Explainability:** The ability to export the KG to a standard format (`.graphml`) and visualize it with tools like `pyvis` and `networkx` is a critical feature. It makes the underlying knowledge base transparent and explorable, which is crucial for building trust and understanding the RAG system's behavior.
