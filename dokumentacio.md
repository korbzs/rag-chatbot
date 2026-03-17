## Agentic RAG Chatbot ([Corrective RAG](https://arxiv.org/pdf/2401.15884) )

Egy lokálisan futó, Corrective RAG (CRAG) architektúrára épülő Agentic chatbot prototípus, amely LangGraph keretrendszert használ. A rendszer autonóm módon dönt a keresési stratégiáról, értékeli a talált kontextus minőségét, és szükség esetén webes kereséssel egészíti ki a hiányzó információkat.

A rendszer kiemelkedő értéke, hogy teljesen ingyenesen üzemeltethető, lokális adatvédelmet garantál (kivéve moderation API + Tavily!!), és képes idegen nyelvű forrásokból is pontos magyar válaszokat generálni.

## Üzemeltetés és Teljesítmény

* **Üzemeltetési költség:** Ingyenes, nyílt forráskódú modellek és Free Tier API-k használata.
* **Hardverigény:** 8-12GB VRAM, 16GB RAM

### Mért Válaszidők (Macbook M1 Pro-n)
A feldolgozási idő függ a kontextus méretétől és a nyelvi fordítás szükségességétől:
* **Angol kérdés -> Angol forrásból:** 20-50 másodperc.
* **Magyar kérdés -> Angol forrásból (szemantikus keresés + fordítás):** 40-80 másodperc.
* **Megbízható Fallback:** Ha a lokális adatbázis nem tartalmazza a választ, a rendszer a Tavily API integrációjának köszönhetően rendkívül releváns és friss információkat ad vissza a webről.

###  Tech Stack & Követelmények

* **Keretrendszer:** LangGraph, LangChain
* **Vektordb:** ChromaDB
* **Embeddings:** Ollama és HuggingFace (bge-m3 / paraphrase-multilingual-MiniLM-L12-v2)
* **LLM:** Ollama, Google Gemini (fallback)
* **Web Search:** Tavily API (free tier)
* **Dokumentum feldolgozás:** PyMuPDF4LLM (PDF -> Markdown) # ehhez egy jobb alternatíva a LlamaParse -> jobban kezeli táblázatokat, latexet
* **Biztonság:** OpenAI Moderation API (free), Regex prompt injection filter
* **Monitorozás:** LangSmith monitoring és log fájlok  

### Fájlstruktúra
```
rag-chatbot/
├── chat_history/          # Felhasználói munkamenetek és memóriák tárolója (JSON)
├── chroma_db/             # Perzisztens vektoradatbázis
├── documents/             # RAG forrásfájlok mappája
│   ├── en/                # Angol nyelvű PDF dokumentumok
│   ├── hu/                # Magyar nyelvű PDF dokumentumok
│   └── md_temp/           # Gyorsítótárazott (cache-elt) Markdown konverziók
├── src/                   # A rendszer logikai magja
│   ├── graph/             # LangGraph state és workflow definíciók
│   ├── nodes/             # Ágens logikák (Guardrail, Router, Retriever, Grader, stb.)
│   ├── rag/               # Dokumentum beolvasás és chunkolás (ingestion.py)
│   ├── security/          # Prompt injection szűrő és OpenAI Moderation API
│   ├── tests/             # Komponens és integrációs tesztek
│   └── tools/             # Külső eszközök, pl. Tavily kereső (search.py)
├── config.py              # Globális konfigurációs változók és System Promptok
├── factory.py             # LLM és Embedding modell példányosító (Factory pattern)
├── main.ipynb             # Fő belépési pont és interaktív tesztelési felület
└── requirements.txt       # Python függőségek listája
```

###  Kísérleti Környezet

A tesztelés és fejlesztés egy Macbook Pro M1 Pro gépen zajlott.
Kipróbált modellek:
* **gemini-2.0-flash:** Jó eredmények, de a Google AI Pro Free Tier limitációi (HTTP 429) miatt a hangsúly a lokális modellekre került.
* **gemma3:12b:** A legjobb átfogó teljesítmény és magyar nyelvi megértés.
* **llama3.2:3b:** Kísérlet a gyorsabb útválasztásra (routing/grading), de a lokális memóriacserék (model-swapping) miatt végül elvetve.


###  Architektúra (LangGraph Workflow)

A rendszer a Corrective RAG irányelveit követi:

```text
 [Kérés] -> [Guardrail] (Biztonsági szűrés)
                 │
                 ▼
             [Router] (Útválasztó ágens)
                 │
      ┌──────────┼──────────┐
 [Vector DB]  [Both]   [Web Search]
      │          │          │
      ▼          ▼          │
[Retriever] ────▶┼──────────┘
                 │
                 ▼
        [Relevance Grader] (Kritikus értékelő ágens)
                 │
         Sikeres?├───────────────── Sikertelen?
                 │                      │
                 ▼                      ▼
            [Generate]             [Rewriter] (Keresés optimalizálása)
            (Válaszadás)                │
                                        ▼
                                  [Web Search] (Újrapróbálkozás, max 3x)
                                        │
                                        └─▶ [Relevance Grader]
```

### [Megvalósítás](images/workflow_graph.png)

- Guardrail: Kiszűri a prompt injection kísérleteket és a nem megfelelő tartalmakat (OpenAI moderation).

- Router: A kérdés típusa alapján dönt a forrásról (vectorstore, web_search, both).

- Retriever: Szemantikus keresést végez a ChromaDB-ben (bge-m3 jobb teljesítménnyel).

- Relevance Grader: LLM alapon értékeli, hogy a talált kontextus teljesen megválaszolja-e a kérdést. Ha nem, új webes keresést triggerel.

- Rewriter: Átfogalmazza a kérdést egy optimális Google/Tavily keresési kifejezéssé.

- Generate: A validált kontextus alapján, a forrásokat megjelölve (citation) generálja a végső választ.

### Tervezési Döntések (Kísérletek és Tanulságok)
**Több kisebb modell vs. Egy nagy modell**: Kipróbáltam, hogy a logikai döntéseket (Router, Grader) egy gyors 3B modell (Llama 3.2), a szövegírást pedig a 12B modell (Gemma 3) végezze. Elvetve: Ollama esetén a VRAM-ba történő folyamatos modell-csere (swapping) drasztikusan lelassította a pipeline-t. Egyetlen erős 12B modell használata gyorsabbnak bizonyult.
**Szekvenciális vs. Párhuzamos végrehajtás**: A dokumentumok aszinkron összefűzése (operator.add) típus- és hash hibákhoz vezetett a LangGraph state-ben. Megoldás: A both útvonalnál egy tiszta szekvenciális pipeline-t alakítottam ki (Retriever -> Web Search), amely biztonságosan fűzi össze a kontextust.
**Hallucináció-kezelés (Temperature)**: A lokális LLM-ek (még 0.0 hőmérsékleten is) megpróbálták memóriából kiegészíteni a hiányzó verseket/himnuszokat, ha a web scraper csak töredéket hozott be. Megoldás: Szigorú Anti-Completion szabály a System Promptban, és a teljes kontextus átadása a generátornak.
**Nyelvi sodródás (Language Drift)**: Angol kérdés esetén a magyar nyelvű kontextus "magyarra húzta" a modellt. Megoldás: Dinamikus nyelv-detektálás a kérdésből, és a válasz nyelvének kőkemény kikényszerítése a végső Human promptban.
**Web scraping korlátok**: A Tavily API csak rövid részleteket ad vissza free tier-ben, ami teljes szövegek (pl. Himnusz) kinyerésére alkalmatlan. Tervezett megoldás: Jina AI link alapján megadja az egész oldalt HTML-ben -> akár regex kulcsszavak alapján -> LLM megkapja az adott match helye +/- 2000 karakter.

### Telepítés és Futtatás

**Virtuális környezet és függőségek**:
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
**Környezeti változók**:
```bash
GEMINI_API_KEY=<api_key>
TAVILY_API_KEY=<api_key>
TAVILY_MCP_LINK=<mcp_link>
OPENAI_API_KEY=<api_key>

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGSMITH_API_KEY=<api_key>
LANGSMITH_PROJECT="rag-chatbot"
```

**Lokális modellek/embeddings (mindegyik külön terminálban)**:
```bash
ollama run gemma3:12b
ollama run llama3.2:3b

ollama pull bge-m3:latest 
```
**Dokumentumok betöltése (Ingestion)**:
Helyezd a kívánt pdf fájlokat a documents/ megfelelő mappájába (hu - magyar, en - angol) és futtasd:
```bash
python src/rag/ingestion.py
```
_(A rendszer automatikusan cache-eli a Markdown konverziót. Újraolvasás kényszerítéséhez használd a --rerun flaget)._

**Chat indítása**:
Jelenleg a belépési pont a main.ipynb.

### Fejlesztési lehetőségek:
  - **Multi-turn Memória és Session Kezelés**: A jelenlegi egykörös (single-turn) architektúra kiterjesztése LangGraph `MemorySaver` vagy perzisztens adatbázis (pl. SQLite/PostgreSQL) bevonásával, hogy a chatbot képes legyen komplex, egymásra épülő kérdések megválaszolására egy hosszabb beszélgetés során (Contextual Memory). Timestamp a chat_history-hoz!
  - **API Végpont és Felhasználói Felület (GUI)**: A logikai mag köré egy REST API építése (pl. FastAPI), amelyre könnyedén ráültethető egy OpenWebUI, Streamlit felület, vagy egy Telegram bot a kényelmesebb hozzáférésért.
  - **100% Lokális Biztonsági Modell (Llama-Guard / ShieldGemma)**: Az OpenAI Moderation API leváltása egy dedikált, lokálisan futó biztonsági LLM-mel. Ezzel a rendszer adatvédelmi szempontból (Privacy) teljesen offline maradhat, kiküszöbölve a harmadik félnek történő adatküldést, miközben robusztusabb prompt injection szűrést kap.
  - **Automatizált RAG Értékelés (RAGAS)**: Egy RAGAS (RAG Assessment) keretrendszer beépítése, amely objektív metrikákkal (pl. Faithfulness, Answer Relevance) méri a találatok és a generált válaszok minőségét, folyamatos és mérhető visszajelzést adva a rendszer pontosságáról a jövőbeli finomhangolásokhoz.
  - **Kód Refaktorálása**: Az egyre növekvő src/nodes/agents.py fájl funkció szerinti szétbontása egy dedikált agents/ könyvtárba (pl. retriever_agent.py, grader_agent.py), javítva a kód átláthatóságát.
  - **Streaming a Generálásnál**: Az érzékelt válaszidő drasztikus csökkentése érdekében a végső generálásnál (Generate node) a folyamatos szövegkiíratás (streaming output) implementálása a backend és a frontend között.
  - **Hybrid Search**: A ChromaDB jelenleg csak Dense (szemantikus) keresést végez. BM25 (Sparse) keresés beépítésével a pontos kulcsszavas keresés (pl. nevek, törvénycikkek) jelentősen javítható lenne.
  - **Semantic Caching**: A gyakran ismétlődő kérdések kiszolgálásának gyorsítása egy Redis vagy LangChain SemanticCache beépítésével (nem kellene a gráfon végigmenni, ha már ismerjük a választ).
  - **Párhuzamosítás**: asyncio használata web search és retriever esetén.
  - **Konténerizáció**: Egy egyszerű docker compose up paranccsal fusson az egész. (main.ipynb jó gyors prototipizálásra).
  - **Több tesztelés**: Integrációs és unittesztek.

### Tesztelés és Monitorozás

A prototípus tesztelése és finomhangolása empirikus módszerekkel történt:

* **Izolált tesztek:** A lokális vektoradatbázis (ChromaDB) és a webes kereső (Tavily API) működését először külön-külön validáltam.
* **Integrált E2E tesztelés:** A teljes LangGraph workflow összehangolt működését a `main.ipynb` notebookban, interaktív módon teszteltem valós kérdésekkel.
* **LangSmith Monitorozás:** A hálózat optimalizálása a LangSmith platform segítségével történt. A futási idők és az ágensek közötti állapotátadások (state) folyamatos nyomon követésével sikerült azonosítani a szűk keresztmetszeteket és felgyorsítani a workflow-t.
