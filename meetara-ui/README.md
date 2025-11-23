# 🎨 Meetara UI - Frontend Interface

A beautiful, modern React frontend for the Meetara AI assistant with RAG capabilities.

## ✨ Features

- **💬 Real-time Chat Interface** - Chat with Meetara AI assistant
- **📁 Domain-specific Document Upload** - Upload documents to specific domains
- **🔍 RAG Status Indicators** - Visual feedback showing when RAG vs LLM is used
- **📊 Domain Management** - View document counts per domain
- **🎯 Session Management** - Maintains conversation context
- **📱 Responsive Design** - Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- Meetara backend running on `http://localhost:8000`

### Installation

```bash
# Navigate to the UI directory
cd meetara-ui

# Install dependencies
npm install

# Start development server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

## 🎯 Key Features Explained

### RAG Status Indicators

The UI shows three different statuses for each response:

- 🟢 **RAG (Green)**: Using retrieved documents from vector store
- 🟡 **LLM (Yellow)**: Using general knowledge when no relevant docs found
- 🔵 **Mixed (Blue)**: Using both documents and general knowledge

### Domain Management

- **Sidebar shows all domains** with document counts
- **Upload documents** to specific domains
- **Real-time updates** when documents are added

### Chat Interface

- **Modern message bubbles** with timestamps
- **Typing indicators** while Meetara is thinking
- **Session persistence** maintains conversation context
- **Copy/share functionality** for responses

## 🛠️ Tech Stack

- **Next.js 14** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **Lucide React** - Beautiful icons
- **Radix UI** - Accessible components

## 📁 Project Structure

```
meetara-ui/
├── src/
│   ├── app/
│   │   ├── page.tsx          # Main chat interface
│   │   ├── layout.tsx        # Root layout
│   │   └── globals.css       # Global styles
│   └── components/           # Reusable components
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

## 🔗 API Integration

The frontend connects to these Meetara backend endpoints:

- `POST /api/chat/` - Send messages to Meetara
- `GET /api/chat/domains` - Get available domains
- `POST /api/upload/doc` - Upload documents
- `GET /api/vectorstore/{domain}` - Get domain status

## 🎨 UI Components

### ChatInterface
- Main chat component with message history
- Real-time RAG status indicators
- Domain detection display

### DocumentUpload
- Drag & drop file upload
- Domain selection
- Progress indicators

### RAGStatus
- Visual indicators for RAG vs LLM
- Document count display
- Similarity score visualization

## 🚀 Deployment

### Build for Production

```bash
npm run build
npm start
```

### Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```

## 🎯 Usage Examples

### Upload Documents
1. Click "Upload Documents" in sidebar
2. Select domain (e.g., "mental_health")
3. Choose PDF/DOC/TXT file
4. Upload and wait for processing

### Chat with RAG
1. Type a question (e.g., "How to manage stress?")
2. Watch for RAG status indicator:
   - 🟢 Green = Found relevant documents
   - 🟡 Yellow = Using general knowledge
3. Get intelligent, contextual responses

### Monitor Domains
- Sidebar shows document counts per domain
- Green = Has documents, Gray = Empty
- Real-time updates after uploads

## 🔧 Customization

### Adding New Domains
Domains are automatically loaded from the backend. No frontend changes needed.

### Styling
Modify `tailwind.config.js` and `globals.css` for custom styling.

### API Endpoints
Update API calls in `page.tsx` if backend endpoints change.

## 🐛 Troubleshooting

### Common Issues

1. **Backend not responding**
   - Ensure Meetara backend is running on port 8000
   - Check CORS settings in backend

2. **Upload fails**
   - Check file format (PDF, DOC, TXT supported)
   - Verify domain exists in backend

3. **RAG status not showing**
   - Check browser console for errors
   - Verify API responses include domain info

## 📈 Performance

- **Lazy loading** for large document lists
- **Debounced search** for better UX
- **Optimized re-renders** with React hooks
- **Efficient API calls** with proper error handling

---

**Ready to chat with Meetara! 🚀**
