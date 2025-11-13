# Training UI - Quick Reference Guide

## Authentication

```javascript
// 1. Get Token
POST /api/auth/token-by-email
Body: { "email": "user@example.com" }
Response: { "success": true, "data": { "token": "..." } }

// 2. Use Token
Headers: { "Authorization": "Bearer <token>" }
```

## Core Workflow

```
1. Login           → POST /api/auth/token-by-email
2. Generate Names  → GET  /api/excel/check-excel?country=Serbia
3. Process Queue   → GET  /api/excel/process (repeat)
4. Sync to Prod    → POST /sync-faces
5. Test Recognition → POST /recognize
```

## Essential Endpoints

| Endpoint | Method | Purpose | Response Time |
|----------|--------|---------|---------------|
| `/api/auth/token-by-email` | POST | Get auth token | < 1s |
| `/api/excel/check-excel?country=X` | GET | Generate celebrity names | 30-60s |
| `/api/excel/process` | GET | Process next person (download images) | 5-15s |
| `/recognize` | POST | Test face recognition | 2-4s |
| `/api/test/recognize` | POST | A/B test both systems | 4-6s |
| `/sync-faces` | POST/GET | Sync to production | Async |
| `/admin/name-mappings` | GET | Get name mappings | < 1s |
| `/manage-image` | POST | Delete/edit image | < 1s |

## Request Examples

### Generate Names
```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:5000/api/excel/check-excel?country=Serbia"
```

### Process Next Person
```bash
curl -H "Authorization: Bearer <token>" \
  "http://localhost:5000/api/excel/process"
```

### Test Recognition
```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -F "image=@test.jpg" \
  "http://localhost:5000/recognize"
```

### A/B Test Recognition
```bash
curl -X POST \
  -H "Authorization: Bearer <token>" \
  -F "image=@test.jpg" \
  -F "ground_truth=John Doe" \
  "http://localhost:5000/api/test/recognize"
```

## Response Patterns

### Success
```json
{
  "success": true,
  "data": { ... }
}
```

### Error
```json
{
  "success": false,
  "error": "Error message"
}
```

### Async (202)
```json
{
  "status": "success",
  "message": "Processing started in background"
}
```

## Key Data Flows

### Flow 1: Generate & Process
```
User → Generate Names (country) → Backend calls OpenAI
     → 50 names added to queue

User → Process Next → Backend searches images (SERP)
     → Downloads 70 images
     → Background: DeepFace validates (2-5 min)
     → Saves 20-40 validated images
```

### Flow 2: Test Recognition
```
User → Upload Image → Backend runs DeepFace
     → Compares against database
     → Returns: person, confidence, match details
```

### Flow 3: A/B Testing
```
User → Upload Image → Backend runs BOTH systems
     → Pipeline A: VGG-Face, threshold 0.35
     → Pipeline B: Facenet512, threshold 0.40
     → Returns: comparison, recommendation
```

## Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Display data |
| 202 | Accepted (async) | Show "Processing..." |
| 400 | Bad request | Show validation error |
| 401 | Unauthorized | Redirect to login |
| 404 | Not found | Show "Not found" |
| 500 | Server error | Show "Try again" |

## Required UI Pages

1. **Login** - Email input, get token
2. **Dashboard** - Stats, quick actions
3. **Generate Names** - Country selector, generate button
4. **Queue Manager** - Process next/all, view queue
5. **Training Progress** - View folders, image counts
6. **Sync** - Move to production
7. **Testing** - Upload & test recognition
8. **Admin** - Name mappings, settings

## Polling Strategy

| What | Interval | Endpoint (to be added) |
|------|----------|------------------------|
| Queue status | 30s | `/api/training/queue-status` |
| Folder status | 10s (when processing) | `/api/training/folder-status` |
| Background jobs | 5s | `/api/training/job-status/:id` |

## File Upload Format

```javascript
const formData = new FormData();
formData.append('image', file);
formData.append('person', 'John Doe');
formData.append('created_date', '2025-01-13');

fetch('/upload-with-domain', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${token}`
    // Don't set Content-Type - browser sets it for FormData
  },
  body: formData
})
```

## Environment Setup

**.env**
```bash
VITE_API_BASE_URL=http://localhost:5000
VITE_API_TIMEOUT=30000
```

**package.json** (React)
```json
{
  "dependencies": {
    "axios": "^1.6.0",
    "react-query": "^3.39.0",
    "react-dropzone": "^14.2.3"
  }
}
```

## Common Pitfalls

❌ **Don't:**
- Include `Content-Type` header with FormData
- Forget Authorization header
- Poll too frequently (< 5s)
- Process without checking queue first

✅ **Do:**
- Store token securely (localStorage)
- Handle 401 by redirecting to login
- Show loading states
- Display progress for async operations
- Handle errors gracefully

## Debug Checklist

- [ ] Token is valid and not expired
- [ ] Authorization header is present
- [ ] Request Content-Type is correct
- [ ] File size < 30MB
- [ ] Image format is supported (JPG, PNG, etc.)
- [ ] Network connection is stable
- [ ] Backend server is running (port 5000)

## Quick Copy-Paste

### Axios Setup
```javascript
import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:5000',
  timeout: 30000
});

api.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;
```

### React Hook - Generate Names
```javascript
import { useState } from 'react';
import api from './api';

export function useGenerateNames() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const generate = async (country) => {
    setLoading(true);
    setError(null);

    try {
      const { data } = await api.get('/api/excel/check-excel', {
        params: { country }
      });

      if (!data.success) {
        throw new Error(data.error || data.message);
      }

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return { generate, loading, error };
}
```

### React Hook - Process Queue
```javascript
import { useState } from 'react';
import api from './api';

export function useProcessQueue() {
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);

  const processNext = async () => {
    setProcessing(true);
    setError(null);

    try {
      const { data } = await api.get('/api/excel/process');

      if (!data.success) {
        throw new Error(data.message);
      }

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setProcessing(false);
    }
  };

  return { processNext, processing, error };
}
```

### React Hook - Test Recognition
```javascript
import { useState } from 'react';
import api from './api';

export function useTestRecognition() {
  const [testing, setTesting] = useState(false);
  const [error, setError] = useState(null);

  const test = async (file, mode = 'single') => {
    setTesting(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const endpoint = mode === 'ab'
        ? '/api/test/recognize'
        : '/recognize';

      const { data } = await api.post(endpoint, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      return data;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setTesting(false);
    }
  };

  return { test, testing, error };
}
```

## Testing Tips

1. **Start with authentication** - verify token works
2. **Test one person** - use `/api/excel/process` once
3. **Check background process** - wait 5 minutes, check folder
4. **Test recognition** - upload test image
5. **Try A/B testing** - compare both systems

## Expected Timings

| Operation | Time | Notes |
|-----------|------|-------|
| Login | < 1s | Instant |
| Generate Names | 30-60s | Per occupation |
| Process Person (API) | 5-15s | Image download |
| Process Person (Full) | 2-5 min | Background validation |
| Test Recognition | 2-4s | Per image |
| A/B Test | 4-6s | Runs both systems |

## Server Logs Location

```bash
# If you need to debug backend
docker logs -f <container-name>
# or
tail -f logs/app.log
```

## Need More Info?

- **Full Documentation**: `TRAINING_UI_API_DOCS.md`
- **System Architecture**: `TRAINING_DATA_COLLECTION_GUIDE.md`
- **A/B Testing**: `AB_TESTING_PLAN.md`
