# Frontend Developer - Getting Started

Welcome! This README will guide you through building the Training UI for the Face Recognition system.

## 📚 Documentation Overview

You have **3 main documents** to work with:

### 1. **TRAINING_UI_API_DOCS.md** ⭐ START HERE
**The complete API documentation** (60+ pages)
- All endpoints with examples
- Request/response formats
- Authentication flow
- Data models
- React/Vue code examples
- Error handling
- Testing guide

### 2. **TRAINING_UI_QUICK_REFERENCE.md** 📖 QUICK LOOKUP
**Quick reference guide** (condensed)
- Endpoint cheat sheet
- Code snippets
- Common patterns
- React hooks ready to copy-paste

### 3. **TRAINING_UI_POSTMAN_COLLECTION.json** 🧪 API TESTING
**Postman collection** (import & test)
- Pre-configured requests
- All endpoints ready to test
- Environment variables setup

## 🚀 Quick Start (5 Minutes)

### Step 1: Import Postman Collection
```bash
1. Open Postman
2. Import → Upload Files
3. Select TRAINING_UI_POSTMAN_COLLECTION.json
4. Set base_url variable to http://localhost:5000
```

### Step 2: Get Authentication Token
```bash
1. Open "Authentication" folder
2. Run "Get Token by Email"
3. Body: { "email": "your-test-email@example.com" }
4. Copy token from response
5. Set auth_token variable in Postman
```

### Step 3: Test Basic Flow
```bash
1. Run "Generate Celebrity Names" (country=Serbia)
2. Wait 30-60 seconds
3. Run "Process Next Person"
4. Check response - should show downloaded images
```

✅ **If all 3 steps work, you're ready to build the UI!**

## 🎯 What You're Building

A web UI for managing face recognition training data with these features:

### Core Features
1. **Login Page** - Email → Get Token
2. **Dashboard** - Overview of training status
3. **Name Generator** - Generate celebrity lists by country
4. **Queue Manager** - Process people one-by-one or in batch
5. **Progress Monitor** - View training folders & image counts
6. **Sync Manager** - Move validated data to production
7. **Testing Page** - Upload images & test recognition
8. **A/B Testing Page** - Compare current vs improved systems

### User Flow
```
Login → Generate Names → Process Queue → Monitor Progress → Sync to Prod → Test
```

## 📋 Required Pages/Components

### 1. Login (Priority: HIGH)
- Email input
- "Get Token" button
- Store token in localStorage
- Redirect to dashboard on success

### 2. Dashboard (Priority: HIGH)
- Stats: Queue size, processed count
- Quick actions: Generate, Process, Sync
- Recent activity

### 3. Generate Names (Priority: HIGH)
- Country input/selector
- "Generate" button
- Loading spinner
- Success message with count

### 4. Process Queue (Priority: HIGH)
- "Process Next" button
- "Process All" button (with pause)
- Current person display
- Progress bar
- Image download stats

### 5. Testing Page (Priority: MEDIUM)
- Image upload (drag & drop)
- Two modes: Single test, A/B test
- Display results with confidence
- Side-by-side comparison for A/B

### 6. Progress Monitor (Priority: MEDIUM)
- List/grid of training folders
- Image counts per person
- Status indicators (ready/processing/insufficient)
- Refresh button

### 7. Sync Manager (Priority: LOW)
- "Sync to Production" button
- Progress indicator
- Success/error messages

### 8. Admin (Priority: LOW)
- Name mappings viewer
- Settings

## 🛠️ Tech Stack Recommendations

### Framework
- **React** (recommended) - Best documented
- **Vue 3** - Also good
- **Next.js** - If you need SSR

### State Management
- **React Query** - For API calls & caching
- **Zustand** - For global state
- **Redux Toolkit** - If you prefer Redux

### UI Library
- **Material-UI (MUI)** - Complete components
- **Ant Design** - Great for admin UIs
- **Chakra UI** - Modern & accessible

### HTTP Client
- **Axios** - Recommended (examples in docs)
- **Fetch API** - Built-in alternative

## 📦 Installation Example (React)

```bash
# Create React app
npx create-react-app training-ui
cd training-ui

# Install dependencies
npm install axios react-query @tanstack/react-query
npm install @mui/material @emotion/react @emotion/styled
npm install react-dropzone react-router-dom

# Start dev server
npm start
```

## 🔑 Environment Setup

Create `.env` file:
```env
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_API_TIMEOUT=30000
```

## 📝 Example Implementation Order

### Week 1: Core Authentication & Flow
1. ✅ Setup project & dependencies
2. ✅ Implement authentication (login page)
3. ✅ Create API client with auth interceptor
4. ✅ Build dashboard (basic stats)
5. ✅ Test all endpoints in Postman

### Week 2: Training Features
6. ✅ Generate names page
7. ✅ Process queue page
8. ✅ Progress monitor
9. ✅ Polish UI & error handling

### Week 3: Testing & Admin
10. ✅ Testing page (single recognition)
11. ✅ A/B testing page
12. ✅ Admin features
13. ✅ Final testing & bug fixes

## 🎨 UI/UX Guidelines

### Loading States
- Always show loading spinners for async operations
- Disable buttons during processing
- Show progress bars for multi-step operations

### Error Handling
- Display user-friendly error messages
- Don't expose technical details
- Provide retry options
- Log errors to console for debugging

### Responsive Design
- Desktop-first (admin tool)
- Minimum width: 1024px
- Tables should scroll horizontally if needed

### Performance
- Lazy load pages/components
- Cache API responses (React Query handles this)
- Debounce search inputs
- Use pagination for large lists

## 🔍 Testing Your Implementation

### Manual Testing Checklist
- [ ] Login with valid email works
- [ ] Login with invalid email shows error
- [ ] Token is stored and used in requests
- [ ] 401 error clears token & redirects to login
- [ ] Generate names shows loading state
- [ ] Process queue displays current person
- [ ] Image upload shows preview
- [ ] Recognition test displays results
- [ ] All error states are handled

### Test Data
**Countries to try:**
- Serbia
- United States
- France
- United Kingdom

**Test Images:**
- Download celebrity photos from Google
- Use clear, frontal face photos
- Test with blurry images (should fail)
- Test with multiple faces (should fail)

## 📞 API Key Notes

### Authentication
- Email-based token system
- Token maps to a domain
- Ask backend team for test emails
- Tokens don't expire (currently)

### Rate Limits
- No hard limits currently
- Be reasonable with polling
- Recommended: 5-10 second intervals

## 🐛 Common Issues & Solutions

### Issue: 401 Unauthorized
**Solution:** Check token is included in Authorization header

### Issue: CORS errors
**Solution:** Backend already configured for CORS. Check you're using correct base URL.

### Issue: File upload fails
**Solution:** Don't set Content-Type header for FormData - let browser set it

### Issue: Long processing times
**Solution:** Expected! Image download = 5-15s, background validation = 2-5 min

### Issue: Can't see progress during background processing
**Solution:** Poll for status updates every 5-10 seconds (or wait for backend to add status endpoint)

## 📚 Additional Resources

### System Architecture
- `TRAINING_DATA_COLLECTION_GUIDE.md` - Complete system overview
- `AB_TESTING_PLAN.md` - A/B testing framework details
- `README.md` - Main project README

### Code Examples
All code examples in `TRAINING_UI_API_DOCS.md` are:
- ✅ Copy-paste ready
- ✅ Production-quality
- ✅ Error-handled
- ✅ TypeScript-ready (types included)

### Design Inspiration
Similar admin UIs:
- AWS Console
- Google Cloud Platform
- Vercel Dashboard
- Strapi Admin Panel

## 💬 Getting Help

1. **Check the docs first** - TRAINING_UI_API_DOCS.md has everything
2. **Test in Postman** - Isolate if it's API or frontend issue
3. **Check backend logs** - For server-side errors
4. **Look at example code** - React hooks & components included

## 🎯 Success Criteria

Your UI is done when:
- [ ] User can login and get token
- [ ] User can generate celebrity names for a country
- [ ] User can process the queue (one or all)
- [ ] User can see training progress
- [ ] User can test recognition with uploaded images
- [ ] User can compare A/B test results
- [ ] All errors are handled gracefully
- [ ] Loading states are shown for async operations

## 🚢 Deployment Notes

### Development
```bash
npm start
# Runs on http://localhost:3000
# API on http://localhost:5000
```

### Production
```bash
npm run build
# Outputs to build/
# Serve with nginx or similar
```

### Environment Variables (Production)
```env
REACT_APP_API_BASE_URL=https://your-api-domain.com
```

## 📌 Important Notes

1. **Background Processing**: When you call `/api/excel/process`, image download is fast (5-15s), but DeepFace validation runs in background (2-5 min). The API returns immediately with download stats.

2. **Polling**: Since there's no WebSocket, poll for status updates:
   - Queue status: Every 30 seconds
   - Processing status: Every 10 seconds (when active)
   - Background jobs: Every 5 seconds

3. **File Uploads**: Max 30MB per file. Supported: JPG, PNG, GIF, BMP, WebP

4. **Authentication**: Token-based, no expiration (currently). Store in localStorage.

5. **API Versioning**: None currently. Base URL is `/api/` for some endpoints, root for others.

## ✨ Bonus Features (Optional)

If you have extra time:
- Dark mode toggle
- Export metrics to CSV
- Bulk image upload
- Advanced filters & search
- Real-time notifications (when backend adds WebSocket)
- Drag & drop queue reordering
- Image preview gallery
- Progress charts/graphs

---

## 🎉 You're Ready!

1. Read `TRAINING_UI_API_DOCS.md` (your bible)
2. Import Postman collection & test endpoints
3. Start with authentication & basic flow
4. Build incrementally
5. Test often

**Good luck! The API is solid and well-documented. You've got this! 🚀**

---

**Questions?** Check the docs first, they're comprehensive!
