# 🧠 Smart Cookbook RAG System & Agentic RAG 🍳📚

Welcome to the **Smart Cookbook RAG System** – a fun, technical, and tasty project that brings together the power of **OpenAI**, **RAG (Retrieval Augmented Generation)**, and **Agentic AI** to transform dusty old cookbooks into dynamic digital culinary assistants! 🤖✨

---

## 🎯 Project Purpose: From Old Recipes to Smart Agents

Have you ever struggled to find the right recipe in a traditional cookbook? This project tackles exactly that using **cutting-edge AI**.

We build:
- 🧾 A system to extract recipe content from scanned cookbooks.
- 🏗️ Structured data suitable for embedding and RAG.
- 🤖 Agentic features to make the system *think* and *act*, not just respond.

---

## 📘 Phase 1: Smart Cookbook RAG 🧑‍🍳

### 🔍 Problem: Cookbooks are Unstructured

📚 Old cookbooks contain loads of valuable recipes, but:
- ❌ No easy search
- 🤷 No contextual filtering
- 🧩 No consistent structure

---

### ⚙️ Solution Architecture

1. 📄 **PDF Input** – Original cookbooks scanned into PDF.
2. 🖼️ **Image Conversion** – PDF pages are turned into JPEGs via `pdf2image`.
3. 🧠 **Text Extraction** – Images sent to **OpenAI GPT-4o** for OCR and content understanding.
4. 🧾 **Structured Formatting** – Extracted data turned into:
   - `title`, `ingredients`, `instructions`, `cuisine`, `dish type`, `tags`
5. 🔎 **Embeddings & RAG Integration** – Processed into vector databases for contextual querying.

> ✅ Example output includes recipes like:
> **Boston Brown Bread**, **Doughnuts**, **Popovers**, **Baked Mackerel**, and more!

---

## 🍽️ Features

- 🧠 **Multi-recipe extraction** across 130+ pages!
- 🔗 **RAG embedding-ready format**
- 📌 Auto-categorization into metadata for tags, cuisine, dish type
- ⚡ Uses **deterministic model prompts** (temperature = 0) for consistent outputs

---

## 🧠 Phase 2: Agentic RAG – Smarter Than Ever 🤖💡

### What is Agentic RAG?

🚀 Taking RAG a step further:
- Instead of just *retrieving* and *responding*...
- Agentic RAG *acts* 🕹️, *decides* 🧠, and *helps* proactively 🤝

---

### ✨ Agentic Abilities

- 🛒 Auto-generate shopping lists
- 📅 Add events to calendars ("Bake cake at 5 PM")
- 🧾 Verify if you have ingredients in your pantry
- 🤝 Work alongside other agents (e.g., fact-checkers, formatters)

---

### 🔄 Continuous Improvement

- 🔁 **Feedback loops** enhance relevance and output
- 📦 **Multi-modal input** (text + image) enables richer interactions
- 🧱 Foundation for future features like:
  - 📸 Visual recipe recognition
  - 📹 Step-by-step video assistants

---

## 🧪 Libraries & Tools

* `openai` – Chat completion + vision APIs
* `pdf2image` – PDF to image conversion
* `base64` – Image encoding
* `dotenv` – Secure API key loading
* `pandas` – Organizing recipe data
* `IPython.display` – Pretty result visualization

---

## 🌍 Applications Beyond Cooking

This system is not just about food 🍲. The same techniques can be applied to:

* 📑 Legal documents
* 🧪 Scientific papers
* 🗣️ Customer reviews
* 🏛️ Historical archives

---

## 📸 Sample Output

```markdown
### 🥖 Recipe Title: Bannocks

**Ingredients**:
- 1 Cupful of Thick Sour Milk
- ½ Cupful of Sugar
- 2 Cupfuls of Flour
- ½ Cupful of Indian Meal
- 1 Teaspoonful of Soda
- A pinch of Salt

**Instructions**:
1. Drop mixture into boiling fat.
2. Serve warm with maple syrup.

**Tags**: 🥞 Breakfast | 🧂 Traditional | 🍁 Sweet
```

---

## 🔐 Setup Instructions

1. Clone this repo
2. Place your `.env` file with `OPENAI_API_KEY`
3. Run the notebook to extract & structure recipes

---

## 🙏 Acknowledgements

This project draws inspiration from the broader work on RAG, Agentic AI, and LangGraph.

Special thanks to:

* 🧑‍🏫 The cooking community for timeless recipes
* 🧠 OpenAI for GPT-4o mini
* 📖 Heritage cookbooks for data

---

## 💡 Future Work

* 📦 Export to JSON/CSV for search integration
* 🧑‍🍳 Web app with visual meal planner
* 🔎 Query recipes by taste, diet, mood, etc.
* 📲 Voice command integration

---

## 🥂 Final Thoughts

By merging the timeless joy of cooking with modern AI, this project not only revives old knowledge but also reimagines how we interact with unstructured data. 🧑‍🍳📈

**Enjoy cooking — the smart way!**

---

