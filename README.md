# Indic Transcript QA & Search System

This project is a multilingual transcript **search and QA system** optimized for **Indic languages**. It uses semantic search with **Qdrant vector store**, **LaBSE embeddings**, and a **RAG (Retrieval-Augmented Generation)** pipeline powered by OpenAI's GPT-4o-mini.

## Features

- Semantic search over user transcripts in **Hindi, Tamil, Telugu, Malayalam**
- **RAG-based question answering** over retrieved transcript chunks
- **Indic embeddings** via `LaBSE` (supports semantic similarity in regional languages)
- **Query rewriting** to standardize informal user queries
- Optimized for **sub-second latency** using `Qdrant`
- REST API with `/search` and `/ask` endpoints
- Sample queries included for testing
- Streamlit frontend for easy exploration

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- Access to [OpenAI API key](https://platform.openai.com/account/api-keys)
- Git, pip

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/transcript-qa.git
cd transcript-qa
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the root directory:

```
OPENAI_API_KEY=your-openai-api-key
```

### 4. Ingest Transcripts

Ensure you have transcript CSV or JSON files under `./transcripts/` folder organized by language.

Then run:

```bash
python ingestion.py
```

This will clean the text, embed transcripts using LaBSE, and index them in Qdrant.

---

## How to Run Locally

### Start FastAPI Backend

```bash
uvicorn main:app --reload
```

- `/search?q=<query>` — Returns top transcript chunks
- `/ask?q=<question>` — RAG-style answer over retrieved transcripts

### Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## Architecture Overview

- **LaBSE Embeddings**: Multilingual sentence-level semantic understanding.
- **Qdrant**: Vector similarity search with filtering on `user_id`, `language`, `timestamp` using regex (for answering and search of transcripts based on user_id, language and time).
- **LangChain + GPT-4o-mini**: For context-aware query rewriting and final answer generation.
- **RAG Prompt**: Strict control over source-based answering, avoids hallucination.
- **Meta Data Augmentation**: Augmented metadata with retrieved transcripts chunks to feed llm (answers questions asking for user_id and date with better semantic retrieval)

---

## Design & Trade-offs

| Component              | Choice                                                                 |
|------------------------|------------------------------------------------------------------------|
| Embeddings             | LaBSE for Indic languages (vs. English translation)                 |
| Vector DB              | Qdrant (metadata capabilities, lightweight, fast, with filtering)                          |
| Query Optimization     | Regex-based filter extractor + GPT-based rewriter + Meta data augmentation after retrieving                   |
| Search Speed           | Sub-second latency (FAISS-like performance with Qdrant)             |
| RAG                    | Prompt with OpenAI GPT-4o-mini via LangChain                 |
| UI                     | Streamlit for quick demo and exploration                            |

---

## Sample Queries & Responses

### 1. Query with Language Filter
**Query:**  
`queries of user id 405`

**Response (Search `/search`)**
```
"Text: எனது கணக்கில் உள்நுழைவதில் எனக்கு சிக்கல் உள்ளது, எனது கடவுச்சொல் மீட்டமைக்கப்படவில்லை.

Metadata: {'user_id': 'user_405', 'language': 'ta', 'timestamp': '2025-05-14T16:30:00', 'source': 'transcripts\tamil\ta1.json', '_id': '92a1496e85d848ca8490fb818d0761bb', '_collection_name': 'transcript_search'}

Text: என் காதலன் என்னிடம் பேசவில்லை, மனதில் துன்பம் உள்ளது.

Metadata: {'user_id': 'user_405', 'language': 'ta', 'timestamp': '2024-05-04T13:30:00', 'source': 'transcripts\tamil\ta2.csv', '_id': '3bc4980eb1014e6b99fa13c709bd14bb', '_collection_name': 'transcript_search'}

"
```

---

### 2. RAG QA
**Query:**  
`What were the top conversation topics among hindi users`

**Response (Ask `/ask`)**
```
{
  "answer": "The top conversation topics among Hindi users included:

1. Customer support and future plans for it (User ID: user_220).
2. Connectivity issues experienced by some users (User ID: user_202).
3. Availability of tutorials for using new functions (User ID: user_215).
4. Online harassment and seeking advice (User ID: user_205).
5. Privacy concerns regarding personal information (User ID: user_109).
6. Changes in updates of the application (User ID: user_206).
7. App performance issues reported by users (User ID: user_213).
8. Return policies for products (User ID: user_212).
9. Personal relationship issues leading to separation (User ID: user_208).
10. Suggestions for adding more options to existing features (User ID: user_214).",


  "sources": "Text: User ID: user_220 , Timestamp: 2025-05-13T12:00:00 , Language: hindi , Transcript: क्या आप भविष्य में हिंदी में और अधिक ग्राहक सहायता प्रदान करने की योजना बना रहे हैं?

Metadata: {'user_id': 'user_220', 'language': 'hi', 'timestamp': '2025-05-13T12:00:00', 'source': 'transcripts\hindi\hi1.json', '_id': '750df56b80f1408fa0583cf39cb6ed30', '_collection_name': 'transcript_search'}

Text: User ID: user_202 , Timestamp: 2025-05-10T10:15:00 , Language: hindi , Transcript: मैंने सुना है कि कुछ उपयोगकर्ताओं को कनेक्टिविटी के मुद्दे आ रहे हैं। क्या यह सच है?

Metadata: {'user_id': 'user_202', 'language': 'hi', 'timestamp': '2025-05-10T10:15:00', 'source': 'transcripts\hindi\hi1.json', '_id': '4331a4e56a68427496a94480757c5876', '_collection_name': 'transcript_search'}

Text: User ID: user_215 , Timestamp: 2025-05-12T13:00:00 , Language: hindi , Transcript: क्या आपके पास कोई ट्यूटोरियल है जो मुझे इस नए फ़ंक्शन का उपयोग करने में मदद कर सके?

Metadata: {'user_id': 'user_215', 'language': 'hi', 'timestamp': '2025-05-12T13:00:00', 'source': 'transcripts\hindi\hi1.json', '_id': '20e1ca94cc9540918c2104afebac4201', '_collection_name': 'transcript_search'}

Text: User ID: user_205 , Timestamp: 2025-05-16T12:45:00 , Language: hindi , Transcript: उसने मुझे ऑनलाइन बहुत हैरेस किया, मैं क्या करूं?

Metadata: {'user_id': 'user_205', 'language': 'hi', 'timestamp': '2025-05-16T12:45:00', 'source': 'transcripts\hindi\hi3.csv', '_id': 'd34728fd4cd9410e8894d43d375b1305', '_collection_name': 'transcript_search'}

Text: User ID: user_109 , Timestamp: 2024-05-05T12:50:00 , Language: hindi , Transcript: मेरे दोस्त ने मेरी निजी बात दूसरों को बता दी।

Metadata: {'user_id': 'user_109', 'language': 'hi', 'timestamp': '2024-05-05T12:50:00', 'source': 'transcripts\hindi\h2.csv', '_id': '71153931d84241b886a617ac9c16c6e2', '_collection_name': 'transcript_search'}

Text: User ID: user_206 , Timestamp: 2025-05-11T14:00:00 , Language: hindi , Transcript: क्या आप मुझे बता सकते हैं कि इस नए अपडेट में क्या बदलाव किए गए हैं?

Metadata: {'user_id': 'user_206', 'language': 'hi', 'timestamp': '2025-05-11T14:00:00', 'source': 'transcripts\hindi\hi1.json', '_id': '472c79faaeb14c0c8a47767fe2913eae', '_collection_name': 'transcript_search'}

Text: User ID: user_213 , Timestamp: 2025-05-12T11:30:00 , Language: hindi , Transcript: कुछ उपयोगकर्ताओं ने बताया कि ऐप बहुत धीमा चल रहा है, क्या आप इसकी जांच कर सकते हैं?

Metadata: {'user_id': 'user_213', 'language': 'hi', 'timestamp': '2025-05-12T11:30:00', 'source': 'transcripts\hindi\hi1.json', '_id': '7b14cbb2025740deaa501b001637f098', '_collection_name': 'transcript_search'}

Text: User ID: user_212 , Timestamp: 2025-05-12T10:45:00 , Language: hindi , Transcript: मैं जानना चाहता हूं कि आपकी वापसी नीति क्या है यदि मैं उत्पाद से संतुष्ट नहीं हूं।

Metadata: {'user_id': 'user_212', 'language': 'hi', 'timestamp': '2025-05-12T10:45:00', 'source': 'transcripts\hindi\hi1.json', '_id': 'eb8a53dfbb3b44f29303b90d91dc9bfe', '_collection_name': 'transcript_search'}

Text: User ID: user_208 , Timestamp: 2025-05-16T13:00:00 , Language: hindi , Transcript: हमारे बीच बहुत लड़ाई होती थी, इसलिए हमने अलग होने का फैसला किया।

Metadata: {'user_id': 'user_208', 'language': 'hi', 'timestamp': '2025-05-16T13:00:00', 'source': 'transcripts\hindi\hi3.csv', '_id': '6d7e096fce5b4418977a47a7da20238a', '_collection_name': 'transcript_search'}

Text: User ID: user_214 , Timestamp: 2025-05-12T12:15:00 , Language: hindi , Transcript: मुझे यह सुविधा बहुत पसंद है, लेकिन क्या आप भविष्य में इसमें और विकल्प जोड़ सकते हैं?

Metadata: {'user_id': 'user_214', 'language': 'hi', 'timestamp': '2025-05-12T12:15:00', 'source': 'transcripts\hindi\hi1.json', '_id': '5a7c3b000b6a453a8eed34d3c286cf46', '_collection_name': 'transcript_search'}

"
}
```

**Query:**  
`malayalam users complaining about design or app issue`

**Response (Ask `/ask`)**
```
{
  "answer": "
  User ID: user_507 - "ഈ ആപ്ലിക്കേഷന്റെ ഡിസൈൻ വളരെ സങ്കീർണ്ണമാണെന്ന് ഞാൻ കരുതുന്നു, ഇത് കൂടുതൽ ലളിതമാക്കണം."
User ID: user_513 - "ചില ഉപയോക്താക്കൾ ആപ്ലിക്കേഷൻ വളരെ സാവധാനത്തിൽ പ്രവർത്തിക്കുന്നുവെന്ന് റിപ്പോർട്ട് ചെയ്തു, നിങ്ങൾക്ക് അത് പരിശോധിക്കാൻ കഴിയുമോ?"
User ID: user_501 - "ഈ ഉൽപ്പന്നത്തിൽ എനിക്ക് ധാരാളം പ്രശ്നങ്ങളുണ്ട്, ഇത് ശരിയായി പ്രവർത്തിക്കുന്നില്ല."
User ID: user_504 - "ഈ ഫീച്ചർ ഒട്ടും ഉപയോഗപ്രദമല്ല, ദയവായി ഇത് പരിഹരിക്കുക അല്ലെങ്കിൽ നീക്കം ചെയ്യുക."
User ID: user_510 - "നിങ്ങളുടെ സേവനത്തിൽ ഞാൻ വളരെ നിരാശനാണ്, നിങ്ങളുടെ ഉപഭോക്തൃ പിന്തുണ ഒട്ടും നല്ലതല്ല."
  ",


  "sources": "Text: User ID: user_518 , Timestamp: 2025-05-16T12:00:00 , Language: malayalam , Transcript: നിങ്ങളുടെ കമ്പനിയിൽ നിന്ന് വന്നതായി തോന്നുന്ന സ്പാം ഇമെയിലുകൾ ലഭിക്കുന്നുണ്ടെന്ന് ചില ഉപയോക്താക്കൾ പരാതിപ്പെട്ടു.

Metadata: {'user_id': 'user_518', 'language': 'ml', 'timestamp': '2025-05-16T12:00:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '80d9a330292d4270b4f84345bcbd3c68', '_collection_name': 'transcript_search'}

Text: User ID: user_513 , Timestamp: 2025-05-15T12:30:00 , Language: malayalam , Transcript: ചില ഉപയോക്താക്കൾ ആപ്ലിക്കേഷൻ വളരെ സാവധാനത്തിൽ പ്രവർത്തിക്കുന്നുവെന്ന് റിപ്പോർട്ട് ചെയ്തു, നിങ്ങൾക്ക് അത് പരിശോധിക്കാൻ കഴിയുമോ?

Metadata: {'user_id': 'user_513', 'language': 'ml', 'timestamp': '2025-05-15T12:30:00', 'source': 'transcripts\malayalam\ml1.json', '_id': 'eeaf76d048d94a19aa13535562eb298e', '_collection_name': 'transcript_search'}

Text: User ID: user_502 , Timestamp: 2025-05-14T14:45:00 , Language: malayalam , Transcript: ചില ഉപയോക്താക്കൾക്ക് കണക്റ്റിവിറ്റി പ്രശ്നങ്ങളുണ്ടെന്ന് ഞാൻ കേട്ടു. ഇത് സത്യമാണോ?

Metadata: {'user_id': 'user_502', 'language': 'ml', 'timestamp': '2025-05-14T14:45:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '432591e4c0f2424486df02ed2dcde537', '_collection_name': 'transcript_search'}

Text: User ID: user_507 , Timestamp: 2025-05-14T18:30:00 , Language: malayalam , Transcript: ഈ ആപ്ലിക്കേഷന്റെ ഡിസൈൻ വളരെ സങ്കീർണ്ണമാണെന്ന് ഞാൻ കരുതുന്നു, ഇത് കൂടുതൽ ലളിതമാക്കണം.

Metadata: {'user_id': 'user_507', 'language': 'ml', 'timestamp': '2025-05-14T18:30:00', 'source': 'transcripts\malayalam\ml1.json', '_id': 'c7e48f41cb884754b2e49c33a02b187c', '_collection_name': 'transcript_search'}

Text: User ID: user_520 , Timestamp: 2025-05-16T13:30:00 , Language: malayalam , Transcript: ഭാവിയിൽ നിങ്ങൾ മലയാളത്തിൽ കൂടുതൽ ഉപഭോക്തൃ പിന്തുണ നൽകാൻ പദ്ധതിയിടുന്നുണ്ടോ?

Metadata: {'user_id': 'user_520', 'language': 'ml', 'timestamp': '2025-05-16T13:30:00', 'source': 'transcripts\malayalam\ml1.json', '_id': 'fdbe0a1f3bec485a93181aa43f9158ad', '_collection_name': 'transcript_search'}

Text: User ID: user_501 , Timestamp: 2025-05-14T14:00:00 , Language: malayalam , Transcript: ഈ ഉൽപ്പന്നത്തിൽ എനിക്ക് ധാരാളം പ്രശ്നങ്ങളുണ്ട്, ഇത് ശരിയായി പ്രവർത്തിക്കുന്നില്ല.

Metadata: {'user_id': 'user_501', 'language': 'ml', 'timestamp': '2025-05-14T14:00:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '01b996d344d146f9b4ebf63d9fef7f7e', '_collection_name': 'transcript_search'}

Text: User ID: user_512 , Timestamp: 2025-05-15T11:45:00 , Language: malayalam , Transcript: ഉൽപ്പന്നത്തിൽ ഞാൻ തൃപ്തനല്ലെങ്കിൽ നിങ്ങളുടെ മടക്കിനൽകൽ നയം എന്താണെന്ന് അറിയാൻ ഞാൻ ആഗ്രഹിക്കുന്നു.

Metadata: {'user_id': 'user_512', 'language': 'ml', 'timestamp': '2025-05-15T11:45:00', 'source': 'transcripts\malayalam\ml1.json', '_id': 'daa1774c6946463cb262dd6e189359c4', '_collection_name': 'transcript_search'}

Text: User ID: user_504 , Timestamp: 2025-05-14T16:15:00 , Language: malayalam , Transcript: ഈ ഫീച്ചർ ഒട്ടും ഉപയോഗപ്രദമല്ല, ദയവായി ഇത് പരിഹരിക്കുക അല്ലെങ്കിൽ നീക്കം ചെയ്യുക.

Metadata: {'user_id': 'user_504', 'language': 'ml', 'timestamp': '2025-05-14T16:15:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '684fa0ada70c4acf922cbf819eaa8001', '_collection_name': 'transcript_search'}

Text: User ID: user_515 , Timestamp: 2025-05-15T14:00:00 , Language: malayalam , Transcript: ഈ പുതിയ ഫംഗ്ഷൻ എങ്ങനെ ഉപയോഗിക്കാമെന്ന് എന്നെ സഹായിക്കുന്ന ഒരു ട്യൂട്ടോറിയൽ നിങ്ങൾക്ക് ഉണ്ടോ?

Metadata: {'user_id': 'user_515', 'language': 'ml', 'timestamp': '2025-05-15T14:00:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '59abcfb3a270481fbe13d42def8d3df4', '_collection_name': 'transcript_search'}

Text: User ID: user_510 , Timestamp: 2025-05-14T20:45:00 , Language: malayalam , Transcript: നിങ്ങളുടെ സേവനത്തിൽ ഞാൻ വളരെ നിരാശനാണ്, നിങ്ങളുടെ ഉപഭോക്തൃ പിന്തുണ ഒട്ടും നല്ലതല്ല.

Metadata: {'user_id': 'user_510', 'language': 'ml', 'timestamp': '2025-05-14T20:45:00', 'source': 'transcripts\malayalam\ml1.json', '_id': '5852b00b596b49b0bc66cc017c40becc', '_collection_name': 'transcript_search'}
"
}
```

---

## Bonus Features Implemented

| Feature                     | Status       | Notes                                                                 |
|-----------------------------|--------------|-----------------------------------------------------------------------|
| Indic Embeddings         | Done       | Using `sentence-transformers/LaBSE`                                   |
| Sub-Second Latency        | Done       | Qdrant + filtered search                                               |
| Query Rewriting           | Done | GPT-powered rewriter for search clarity (can be enabled)              |
| LLM RAG Answering         | Done       | GPT-4o-mini + LangChain                                               |
| Streamlit UI              | Done       | Interface to query `/search` and `/ask`                               |
