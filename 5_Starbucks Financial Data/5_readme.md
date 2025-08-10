## 🚀 Project Analysis: Multimodal RAG for Starbucks Financials

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** system designed to answer complex financial questions about Starbucks. It uniquely processes and understands information from two different sources: **audio recordings** of earnings calls and **PDF documents** of financial releases.

### 🎯 Project Overview

The primary goal is to build a conversational AI bot that acts as a financial analyst. Instead of manually sifting through hours of audio and dozens of pages of reports, a user can simply ask a question and get a concise, data-driven answer.

*   **Data Sources 📊:**
    *   **Audio 🎙️:** An MP3 file of a Starbucks earnings call (`starbucks-q3.mp3`).
    *   **Documents 📄:** A PDF financial report (`3Q24-Earnings-Release.pdf`).
*   **Core Challenge 🧗:** Financial data is dense and spread across different formats. This system tackles that by creating a unified understanding of both text and visual information.
*   **Key Innovation ✨:** The system **independently retrieves relevant information from both the audio transcript and the document's pages (as images)** before synthesizing a final answer.

---

### ⚙️ Technical Workflow & Implementation

The project follows a logical, step-by-step process to build the RAG pipeline. Here’s a breakdown of the implementation seen in the notebook:

#### 1. Environment & Setup 🛠️

The foundation is built using a powerful stack of Python libraries, setting the stage for the entire workflow.
*   **Core AI/ML:** `openai`, `langchain`, `torch`, `sentence-transformers`.
*   **Audio Processing:** `openai-whisper` for state-of-the-art speech-to-text.
*   **Data Handling:** `pdf2image`, `Pillow` (PIL), `pandas`.

The code smartly checks for a **GPU (`cuda`)** to accelerate the demanding tasks of transcription and embedding.

#### 2. Audio Processing: From Speech to Text 🎙️➡️📝

The system first tackles the audio data from the earnings call.
*   **Transcription:** The `whisper` model, specifically the powerful `large-v3-turbo` variant, is used to convert the `starbucks-q3.mp3` audio file into a full text transcript.
*   **Chunking:** The resulting long transcript is too large to be processed at once. It's broken down into smaller, manageable **chunks of 250 characters**. This allows the model to find very specific, relevant snippets later on. In this case, the transcript was divided into **58 chunks**.

```python
# In [13] - Chunking the transcribed audio text
chunk_size = 250
audio_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
```

#### 3. PDF Processing: From Document to Images 📄➡️🖼️

The PDF report isn't treated as text. Instead, it's converted into a series of images, one for each page. This captures all the visual elements like tables, charts, and layout, which are often lost in simple text extraction.
*   **Conversion:** The `pdf2image` library converts the **17-page PDF** into **17 separate JPG images**.
*   **Rationale:** This approach allows the system to "see" the financial tables and charts just as a human would, enabling a richer understanding of the data.

#### 4. Multimodal Embedding with CLIP 🧠

This is the most critical step, where the magic of multimodal AI happens.
*   **The Model:** The `clip-ViT-B-32` model from the `sentence-transformers` library is used. CLIP is designed to understand both text and images in the same "thought space."
*   **Unified Vector Space:** Both the **text chunks** from the audio and the **page images** from the PDF are fed into the CLIP model. It converts each item into a 512-dimensional vector (an embedding).
*   **The Result:**
    *   `audio_embeddings`: Shape `(58, 512)` -> 58 text chunks, each represented by a 512-number vector.
    *   `image_embeddings`: Shape `(17, 512)` -> 17 page-images, each represented by a 512-number vector.

Now, a text query can be directly and mathematically compared to both other texts and images to find the best match.

#### 5. The Retrieval Engine: Finding the Best Clues 🔍

With the data embedded, the system can now retrieve information relevant to a user's query.
*   **Query Embedding:** The user's question (e.g., *"what are the short term risks for the company"*) is also embedded into a 512-dimensional vector using the same CLIP model.
*   **Dual Search:** The system performs two separate searches using **cosine similarity**:
    1.  **Audio Search:** The query vector is compared against all **58 audio chunk vectors**.
    2.  **Image Search:** The query vector is compared against all **17 image vectors**.
*   **Ranking:** The system identifies the top `k` most similar chunks and images. In the example, it retrieves the top 20 audio chunks and the top 5 images.

#### 6. Context Assembly for the LLM 📦

The top search results are collected to serve as the "context" or "evidence" for the final answer.
*   **Text Context:** The text from the top 5 most relevant audio chunks is joined together into a single block of text.
*   **Image Context:** The top 5 most relevant page images are loaded and encoded into **Base64 format**. This is a standard method for sending images to an AI model via an API.

#### 7. The Generative Step: Synthesizing the Answer 🤖

Finally, all the pieces are sent to a large language model (LLM) to generate a human-readable answer.
*   **Model:** `gpt-4o-mini` is used for its powerful reasoning and multimodal capabilities.
*   **Prompting:** A carefully crafted **system prompt** instructs the model to act as a "financial advisor expert" and, crucially, to base its answer **only on the provided context** (the text and images).
*   **Multimodal Input:** The model receives a complex message containing:
    1.  The role-defining system prompt.
    2.  The user's original query.
    3.  The concatenated text context from the audio.
    4.  The Base64-encoded images from the PDF.
*   **Final Output:** The LLM synthesizes all this information to produce a structured, detailed answer, as seen in the final output cell.

### 💡 Key Insights & Takeaways

*   **True Multimodality:** This is a fantastic example of a true multimodal RAG system. It doesn't just process text; it leverages visual data from PDFs, enabling a more comprehensive analysis.
*   **Independent Retrieval:** The strategy of searching audio and images separately and then combining the results is highly effective. It allows the system to draw evidence from the best source for any given query.
*   **Grounded Generation:** The use of a strong system prompt and providing explicit context (the retrieved chunks and images) ensures the model's response is **grounded in the source data**, minimizing the risk of hallucination.
*   **End-to-End Automation:** This project demonstrates a complete, automated pipeline from raw, unstructured data (audio/PDF) to a precise, structured answer, showcasing the immense practical value of modern AI techniques.