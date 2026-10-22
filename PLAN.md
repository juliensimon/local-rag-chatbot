# Plan: Replace Gradio UI with Modern React Web Interface

## Goal
Replace the current Gradio-based UI with a modern React frontend while preserving all existing chatbot features and maintaining the Python backend.

---

## Current Features to Preserve

### Core Chat Features
- Real-time streaming responses (token-by-token)
- Chat history persistence within session
- Clear conversation functionality
- Example questions for quick testing

### RAG Mode Features
- Toggle between RAG and vanilla LLM modes
- Three search types: MMR, Similarity, Hybrid
- Document filter dropdown (filter by specific PDF)
- Retrieved context display with:
  - Source document citations (filename, page number)
  - Relevance/distance scores
  - Top chunk highlighting (star indicator)
  - Rewritten query display (when query rewriting enabled)
  - Hybrid scores breakdown (fused/semantic/keyword)

### Advanced Options
- Query rewriting toggle
- Re-ranking toggle
- Hybrid alpha slider (0-100, semantic vs keyword balance)
- Conditional visibility (hybrid slider only shows when hybrid search selected)

### Security Features
- Path traversal protection for document filters
- Query length truncation (MAX_QUERY_LENGTH)

---

## Architecture

### Recommended Stack
```
Frontend: React 18 + TypeScript + Vite
Styling: Tailwind CSS (or shadcn/ui for polished components)
State: React Context or Zustand (lightweight)
HTTP: Native fetch with EventSource for SSE streaming
Backend: FastAPI (replaces Gradio's built-in server)
```

### Why FastAPI?
- Native SSE (Server-Sent Events) support for streaming
- OpenAPI/Swagger docs auto-generated
- Async support matches well with LangChain streaming
- Clean separation from UI framework

---

## Implementation Steps

### Phase 1: Backend API Layer
Create `api/` directory with FastAPI endpoints that wrap existing logic.

**Files to create:**
- `api/__init__.py`
- `api/main.py` - FastAPI app, CORS, routes
- `api/routes.py` - Endpoint definitions
- `api/schemas.py` - Pydantic models for request/response
- `api/streaming.py` - SSE streaming helpers

**Endpoints:**
```
GET  /api/health              - Health check
GET  /api/sources             - List available document sources
POST /api/chat                - Non-streaming chat (simple)
POST /api/chat/stream         - SSE streaming chat (main endpoint)
```

**POST /api/chat/stream request body:**
```json
{
  "message": "string",
  "history": [{"role": "user|assistant", "content": "string"}],
  "rag_enabled": true,
  "search_type": "mmr|similarity|hybrid",
  "doc_filter": "filename.pdf|null",
  "use_query_rewriting": false,
  "use_reranking": false,
  "hybrid_alpha": 70
}
```

**SSE response events:**
```
event: token
data: {"content": "partial text"}

event: context
data: {"sources": [...], "scores": [...], "rewritten_query": "..."}

event: done
data: {}

event: error
data: {"message": "error description"}
```

### Phase 2: React Frontend Setup
Create `frontend/` directory with Vite + React + TypeScript.

**Directory structure:**
```
frontend/
├── src/
│   ├── components/
│   │   ├── Chat/
│   │   │   ├── ChatContainer.tsx
│   │   │   ├── ChatMessage.tsx
│   │   │   ├── ChatInput.tsx
│   │   │   └── StreamingMessage.tsx
│   │   ├── Controls/
│   │   │   ├── RagToggle.tsx
│   │   │   ├── SearchTypeSelector.tsx
│   │   │   ├── DocumentFilter.tsx
│   │   │   └── AdvancedOptions.tsx
│   │   ├── Context/
│   │   │   ├── ContextPanel.tsx
│   │   │   └── SourceCard.tsx
│   │   └── Layout/
│   │       ├── Header.tsx
│   │       └── ExampleQuestions.tsx
│   ├── hooks/
│   │   ├── useChat.ts          - Chat state management
│   │   ├── useStreaming.ts     - SSE connection handling
│   │   └── useSources.ts       - Document sources fetching
│   ├── types/
│   │   └── index.ts            - TypeScript interfaces
│   ├── api/
│   │   └── client.ts           - API client functions
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── tsconfig.json
├── vite.config.ts
└── tailwind.config.js
```

### Phase 3: Core Components Implementation

**ChatContainer.tsx** - Main chat interface
- Message list with user/assistant styling
- Auto-scroll to bottom on new messages
- Loading state during streaming

**ChatInput.tsx** - Message input
- Textarea with submit button
- Enter to send, Shift+Enter for newline
- Disabled state during streaming

**StreamingMessage.tsx** - Handles in-progress responses
- Displays partial text as it arrives
- Typing indicator/cursor animation
- Smooth text appearance

**ContextPanel.tsx** - Retrieved documents display
- Collapsible panel (shown when RAG enabled)
- Source cards with scores and highlights
- Rewritten query display

**Controls** - RAG configuration
- Toggle switch for RAG mode
- Radio buttons for search type
- Dropdown for document filter
- Collapsible advanced options section

### Phase 4: Streaming Implementation

**useStreaming.ts hook:**
```typescript
function useStreaming() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [partialResponse, setPartialResponse] = useState("");
  const [context, setContext] = useState<Context | null>(null);

  const startStream = async (request: ChatRequest) => {
    setIsStreaming(true);
    setPartialResponse("");

    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      // Parse SSE events and update state
      parseSSEEvents(chunk, {
        onToken: (text) => setPartialResponse(prev => prev + text),
        onContext: (ctx) => setContext(ctx),
        onDone: () => setIsStreaming(false),
        onError: (err) => { /* handle error */ },
      });
    }
  };

  return { isStreaming, partialResponse, context, startStream };
}
```

### Phase 5: Integration & Testing

1. **Development proxy**: Configure Vite to proxy `/api` to FastAPI backend
2. **Environment config**: `.env` files for API URL configuration
3. **Error handling**: Network errors, streaming interruptions, API errors
4. **Loading states**: Skeleton loaders, spinners, disabled states
5. **Responsive design**: Mobile-friendly layout

### Phase 6: Production Build & Deployment

**Option A: Separate services**
- Build React app: `npm run build` → static files
- Serve static files from FastAPI or nginx
- Keep FastAPI as API server

**Option B: Single service (simpler)**
- FastAPI serves both API and static React build
- Single deployment unit
- Add `StaticFiles` mount in FastAPI

---

## Migration Strategy

### Incremental Approach (Recommended)
1. Build FastAPI backend alongside Gradio (both can run)
2. Build React frontend pointing to FastAPI
3. Test thoroughly in parallel
4. Switch over when ready
5. Remove Gradio code

### File Changes Summary
```
Keep (no changes):
- config.py
- models.py
- qa_chain.py
- retrievers.py
- vectorstore.py
- utils.py (most of it)
- cli.py
- tests/ (add API tests)

Deprecate (remove after migration):
- ui/
- app.py (Gradio entry point)

Add:
- api/
- frontend/
- app_api.py (new FastAPI entry point)
```

---

## Key Implementation Details

### SSE Streaming Format
```python
# api/streaming.py
async def stream_response(qa_chain, request):
    async def event_generator():
        for chunk_data in qa_chain.stream(inputs):
            # Send token
            yield f"event: token\ndata: {json.dumps({'content': chunk_data['chunk']})}\n\n"

        # Send context at end
        yield f"event: context\ndata: {json.dumps(format_context(chunk_data))}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

### TypeScript Interfaces
```typescript
// types/index.ts
interface Message {
  role: "user" | "assistant";
  content: string;
}

interface ChatRequest {
  message: string;
  history: Message[];
  rag_enabled: boolean;
  search_type: "mmr" | "similarity" | "hybrid";
  doc_filter: string | null;
  use_query_rewriting: boolean;
  use_reranking: boolean;
  hybrid_alpha: number;
}

interface SourceDocument {
  content: string;
  source: string;
  page: number;
  score?: number;
  semantic_score?: number;
  keyword_score?: number;
  is_top?: boolean;
}

interface Context {
  sources: SourceDocument[];
  rewritten_query?: string;
}
```

---

## Estimated Effort

| Phase | Description | Complexity |
|-------|-------------|------------|
| 1 | FastAPI backend | Medium |
| 2 | React project setup | Low |
| 3 | Core components | Medium |
| 4 | Streaming implementation | Medium-High |
| 5 | Integration & testing | Medium |
| 6 | Production build | Low |

---

## Questions to Consider

1. **Styling framework**: Tailwind CSS, shadcn/ui, Material UI, or custom?
2. **State management**: React Context sufficient, or use Zustand/Redux?
3. **Deployment target**: Same as current (HuggingFace Spaces), or different?
4. **Authentication**: Add user auth, or keep anonymous?
5. **Persistence**: Add chat history persistence (localStorage, database)?

---

## Alternative Approaches

### Option B: Keep Gradio, Customize More
- Use Gradio Blocks with heavy custom CSS/JS
- Less work but limited flexibility
- Still constrained by Gradio's patterns

### Option C: Next.js Full-Stack
- Server components + API routes in one framework
- Good if deploying to Vercel
- More opinionated, steeper learning curve

### Option D: htmx + Jinja Templates
- Simpler, no JavaScript framework
- Server-rendered with dynamic updates
- Good for simpler UIs, less suitable for real-time streaming

**Recommendation**: Option A (React + FastAPI) provides the best balance of flexibility, maintainability, and modern developer experience.
