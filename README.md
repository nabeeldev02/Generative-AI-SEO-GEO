# Generative-AI-SEO-GEO

A complete backend pipeline that automatically generates **SEO-optimized articles** using SERP analysis, reranking, and LLM-powered content creation (Gemini / OpenAI).

This tool fetches SERP results, analyzes top-ranking pages, generates an outline, writes a full article, adds SEO metadata, and produces a final JSON output.

---

## **✨ Features**

* 🔍 **SERP fetching** via SerpAPI
* 📊 **Relevance scoring & reranking** using SentenceTransformer
* 🧠 *LLM-powered* outline + content generation
* 🏷️ SEO metadata generation (Title Tag, Meta Description)
* ❓ Automatic FAQ generation
* 📈 SEO scoring
* 🗃️ Final article saved as `article_<topic>.json`

---

## **📦 Installation**

### 1️⃣ Clone your project

```bash
git clone https://github.com/nabeeldev02/Generative-AI-SEO-GEO.git
cd Article_Generation
```

---

## **🔧 Create & Activate Virtual Environment**

### **Linux / macOS**

```bash
python3 -m venv myenv
source myenv/bin/activate
```

### **Windows**

```bash
python -m venv myenv
myenv\Scripts\activate
```

---

## **📥 Install Dependencies**

Install all required packages:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:

```bash
pip install sentence-transformers openai google-generativeai requests pydantic
```

---

## **🔑 Set API Keys**
Please insert the api in the code.

## **🚀 Run the Application**

Just run:

```bash
python main.py
```

You will be asked:

* Topic / keyword
* Language
* AI provider (gemini/openai)

Example:

```
📝 Enter topic/keyword: best productivity tools for remote teams
🌍 Language: English
🤖 AI Provider: gemini
```

---

## **📄 Output**

When completed:

* SEO Score
* Title Tag + Meta Description
* Primary keywords
* Internal & External links
* FAQ count

Final full article is saved as:

```
article_best_productivity_tools_for_remote_teams.json
```

---

## **📁 Project Structure**

```
.
├── main.py
├── README.md
├── requirements.txt
└── article_<topic>.json     # generated output
```
