# Face Recognition Training UI - Frontend Developer Documentation

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [API Endpoints](#api-endpoints)
4. [Data Models](#data-models)
5. [Workflow & UI Requirements](#workflow--ui-requirements)
6. [Example Code](#example-code)
7. [Error Handling](#error-handling)
8. [Testing](#testing)

---

## Overview

### Purpose
Build a web UI for the automated training data collection system that allows users to:
- Generate lists of celebrities by occupation and country
- Trigger image collection and processing
- Monitor processing status
- View and manage collected training data
- Track training progress and statistics

### Architecture
```
┌─────────────────┐
│   Frontend UI   │ ← You build this
│   (React/Vue)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│  Flask Backend  │ ← Already exists
│   (Python API)  │
└─────────────────┘
```

### Base URL
```
Development: http://localhost:5000
Production: <Your production URL>
```

### Technology Stack Recommendations
- **Framework**: React, Vue.js, or Next.js
- **HTTP Client**: Axios or Fetch API
- **State Management**: Redux, Zustand, or React Context
- **UI Components**: Material-UI, Ant Design, or Chakra UI
- **Real-time Updates**: Polling or WebSockets (polling for now)

---

## Authentication

### How Authentication Works

The API uses **Bearer Token authentication**. Each client has a unique token that maps to a domain.

### Getting a Token

#### Endpoint: `POST /api/auth/token-by-email`

**Request**:
```json
{
  "email": "user@example.com"
}
```

**Response (Single Domain)**:
```json
{
  "success": true,
  "data": {
    "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
    "email": "user@example.com"
  }
}
```

**Response (Multiple Domains)**:
```json
{
  "success": true,
  "data": [
    {
      "token": "dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD",
      "email": "user@example.com",
      "domain": "domain1"
    },
    {
      "token": "anotherToken123456789",
      "email": "user@example.com",
      "domain": "domain2"
    }
  ]
}
```

**Error Responses**:
```json
// 404 - Email not found
{
  "success": false,
  "error": "Email 'unknown@example.com' not found in authorized users"
}

// 400 - Bad request
{
  "success": false,
  "error": "Email field is required"
}
```

### Using the Token

Include the token in the `Authorization` header for all subsequent requests:

```http
Authorization: Bearer dJfY7Aq4mycEYEtaHxAiY6Ok43Me5IT2QwD
```

### Validation Endpoint

#### Endpoint: `POST /api/auth/validate-email`

Use this to check if an email has access without getting the token:

**Request**:
```json
{
  "email": "user@example.com"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "email": "user@example.com",
    "has_access": true,
    "key": "domain1"
  }
}
```

---

## API Endpoints

### 1. Training Data Collection

#### 1.1 Generate Celebrity Names

**Endpoint**: `GET /api/excel/check-excel`

**Purpose**: Generate a list of celebrities for a specific country and occupation using AI

**Query Parameters**:
- `country` (required): Country name (e.g., "Serbia", "United States", "France")

**Headers**:
```http
Authorization: Bearer <token>
```

**Request Example**:
```bash
GET /api/excel/check-excel?country=Serbia
```

**Response (Success)**:
```json
{
  "success": true,
  "message": "Excel file exists and contains 50 rows. Processing started in background for country: Serbia.",
  "row_count": 50,
  "file_path": "storage/excel/occupation.xlsx",
  "thread_started": true,
  "country": "Serbia"
}
```

**Response (Error - File Not Found)**:
```json
{
  "success": false,
  "message": "Excel file not found at path: storage/excel/occupation.xlsx"
}
```

**Response (Error - File Empty)**:
```json
{
  "success": false,
  "message": "Excel file is empty"
}
```

**What Happens**:
1. Backend reads occupations from `occupation.xlsx` (Actor, Politician, etc.)
2. For each occupation, calls OpenAI GPT-4.1 to generate 20+ celebrity names
3. Validates names with a second AI call
4. Saves to `data.xlsx` queue
5. Process runs in background (async)

**Processing Time**: ~30-60 seconds per occupation

**UI Considerations**:
- Show loading spinner while processing
- Display success message when complete
- Poll for status updates (see monitoring section)

---

#### 1.2 Process Single Person (Fetch Images)

**Endpoint**: `GET /api/excel/process`

**Purpose**: Process the next person in the queue - search, download, and validate images

**Headers**:
```http
Authorization: Bearer <token>
```

**Request Example**:
```bash
GET /api/excel/process
```

**Response (Success)**:
```json
{
  "success": true,
  "data": {
    "name": "Dragan",
    "last_name": "Bjelogrlic",
    "occupation": "Actor",
    "original_name": "Dragan",
    "original_last_name": "Bjelogrlić",
    "original_occupation": "glumac"
  },
  "images": {
    "success": true,
    "message": "Successfully downloaded 45 images",
    "count": 45,
    "images": [
      {
        "filename": "Dragan_Bjelogrlic_20250113_044003_1.jpg",
        "path": "storage/training/serbia/Dragan_Bjelogrlic_20250113_044003_1.jpg",
        "source_url": "https://example.com/image1.jpg",
        "size": 245678,
        "dimensions": "1200x800"
      }
    ],
    "failed": [],
    "total_found": 100,
    "processed": 70
  }
}
```

**Response (Error - No Data)**:
```json
{
  "success": false,
  "message": "No data found in Excel file or file not found"
}
```

**What Happens**:
1. Reads first person from `data.xlsx`
2. Removes that row from the queue
3. Searches RapidAPI for images: "Dragan Bjelogrlic Actor"
4. Downloads up to 70 images
5. **Background process starts** (DeepFace validation):
   - Finds 3 reference images
   - Verifies all other images match those references
   - Saves 20-40 validated faces to `trainingPassSerbia/`
   - Takes 2-5 minutes in background

**Processing Time**:
- API response: ~5-15 seconds
- Background validation: ~2-5 minutes

**UI Considerations**:
- Show immediate feedback when API returns
- Display progress indicators for background processing
- Allow user to trigger multiple in sequence
- Show how many people remain in queue

---

#### 1.3 Upload Manual Images

**Endpoint**: `POST /upload-with-domain`

**Purpose**: Manually upload images for a specific person (alternative to automatic search)

**Headers**:
```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form Data**:
- `image` (file, required): Image file
- `person` (string, required): Person name (e.g., "John Doe")
- `created_date` (string, required): Date in format YYYY-MM-DD

**Request Example**:
```javascript
const formData = new FormData();
formData.append('image', imageFile);
formData.append('person', 'John Doe');
formData.append('created_date', '2025-01-13');

fetch('/upload-with-domain', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer <token>'
  },
  body: formData
})
```

**Response (Success)**:
```json
{
  "status": "success",
  "message": "Image sent for processing"
}
```

**Response (Error)**:
```json
{
  "error": "No image in request"
}
```

**HTTP Status Codes**:
- `202`: Accepted (processing started)
- `400`: Bad request (missing parameters)
- `401`: Unauthorized

---

### 2. Monitoring & Status

#### 2.1 Check Queue Status

**Note**: This endpoint doesn't exist yet, but you'll need it. For now, you can:
- Poll `/api/excel/check-excel?country=<country>` to see `row_count`
- Subtract from original count to show progress

**Suggested Implementation** (Backend team will add):
```
GET /api/training/queue-status
Response:
{
  "total_in_queue": 45,
  "processed": 5,
  "in_progress": 1,
  "failed": 0
}
```

#### 2.2 Check Training Folder Status

**Note**: Currently requires file system access. Suggested endpoint:

```
GET /api/training/folder-status
Response:
{
  "total_people": 120,
  "total_images": 4800,
  "people": [
    {
      "name": "Dragan_Bjelogrlic",
      "image_count": 38,
      "status": "ready",  // ready, processing, insufficient
      "folder_path": "storage/trainingPassSerbia/Dragan_Bjelogrlic"
    }
  ]
}
```

---

### 3. Production Database Management

#### 3.1 Sync Training to Production

**Endpoint**: `POST /sync-faces` or `GET /sync-faces`

**Purpose**: Copy validated faces from training folders to production database

**Query Parameters** (optional):
- `source_dir`: Source directory (default: `storage/recognized_faces`)
- `target_dir`: Target directory (default: `storage/recognized_faces_prod`)

**Headers**:
```http
Authorization: Bearer <token>
```

**Request Example**:
```bash
POST /sync-faces?source_dir=storage/recognized_faces&target_dir=storage/recognized_faces_prod
```

**Response (Success)**:
```json
{
  "status": "success",
  "message": "Sync started in background",
  "source_dir": "storage/recognized_faces",
  "target_dir": "storage/recognized_faces_prod"
}
```

**HTTP Status Code**: `202 Accepted` (background process)

**What Happens**:
1. Copies images from source to target
2. Runs face recognition test
3. Validates the sync was successful
4. Background process (async)

---

### 4. Admin & Management

#### 4.1 Get Name Mappings

**Endpoint**: `GET /admin/name-mappings`

**Purpose**: Get mappings between normalized and original names (handles special characters)

**Response**:
```json
{
  "status": "success",
  "count": 150,
  "mappings": [
    {
      "normalized": "Dragan_Bjelogrlic",
      "original": "Dragan Bjelogrlić"
    },
    {
      "normalized": "Milos_Bikovic",
      "original": "Miloš Biković"
    }
  ]
}
```

#### 4.2 Manage Images

**Endpoint**: `POST /manage-image`

**Purpose**: Edit or delete images from the database

**Headers**:
```http
Authorization: Bearer <token>
Content-Type: application/json
```

**Request (Delete)**:
```json
{
  "filename": "John_Doe_2025-01-13_04400301.jpg",
  "action": "delete"
}
```

**Request (Edit)**:
```json
{
  "filename": "John_Doe_2025-01-13_04400301.jpg",
  "action": "edit",
  "person": "Jane Doe"
}
```

**Response (Success)**:
```json
{
  "status": "success",
  "message": "Image deleted successfully",
  "filename": "John_Doe_2025-01-13_04400301.jpg"
}
```

---

### 5. Testing & Validation

#### 5.1 Test Recognition

**Endpoint**: `POST /recognize`

**Purpose**: Test face recognition on an uploaded image

**Headers**:
```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form Data**:
- `image` (file, required): Image file to test

**Response (Success)**:
```json
{
  "status": "success",
  "person": "Dragan Bjelogrlic",
  "confidence": 95.2,
  "best_match": {
    "identity": "Dragan_Bjelogrlic_2025-01-13_04400305.jpg",
    "distance": 0.28,
    "confidence_metrics": {
      "confidence_percentage": 95.2,
      "quality_score": "high"
    }
  },
  "recognized_persons": [
    {
      "person": "Dragan Bjelogrlic",
      "confidence": 95.2,
      "match_count": 5
    }
  ],
  "api_info": {
    "request_processing_time": 2.34,
    "total_faces_detected": 1,
    "valid_faces_after_filtering": 1
  }
}
```

**Response (No Face Found)**:
```json
{
  "status": "no_faces",
  "message": "No valid faces found after validation checks",
  "recognized_faces": [],
  "total_faces_detected": 0
}
```

#### 5.2 A/B Test Recognition

**Endpoint**: `POST /api/test/recognize`

**Purpose**: Test both current and improved recognition systems side-by-side

**Headers**:
```http
Authorization: Bearer <token>
Content-Type: multipart/form-data
```

**Form Data**:
- `image` (file, required): Image file to test
- `image_id` (string, optional): Identifier for tracking
- `ground_truth` (string, optional): Known correct answer for accuracy testing

**Response**:
```json
{
  "image_id": "test_001",
  "ground_truth": "Dragan Bjelogrlic",
  "pipeline_a_result": {
    "status": "success",
    "person": "Dragan Bjelogrlic",
    "confidence": 93.5,
    "processing_time": 2.14,
    "profile_used": {
      "name": "current",
      "model": "VGG-Face",
      "threshold": 0.35
    }
  },
  "pipeline_b_result": {
    "status": "success",
    "person": "Dragan Bjelogrlic",
    "confidence": 97.2,
    "processing_time": 2.45,
    "profile_used": {
      "name": "improved",
      "model": "Facenet512",
      "threshold": 0.40
    }
  },
  "comparison": {
    "comparison_id": "uuid-here",
    "comparison_metrics": {
      "both_succeeded": true,
      "results_match": true,
      "confidence_difference": 3.7,
      "processing_time_difference": 0.31,
      "faster_pipeline": "pipeline_a"
    }
  },
  "recommendation": "Both agree. Pipeline B has 3.7% higher confidence."
}
```

---

## Data Models

### Person/Celebrity
```typescript
interface Person {
  name: string;
  last_name: string;
  occupation: string;
  original_name?: string;
  original_last_name?: string;
  image_count?: number;
  status?: 'pending' | 'processing' | 'completed' | 'failed';
}
```

### Image
```typescript
interface Image {
  filename: string;
  path: string;
  source_url?: string;
  size: number;
  dimensions: string;
  person: string;
  created_date: string;
}
```

### Training Status
```typescript
interface TrainingStatus {
  total_in_queue: number;
  processed: number;
  in_progress: number;
  failed: number;
  current_person?: Person;
}
```

### Recognition Result
```typescript
interface RecognitionResult {
  status: 'success' | 'no_faces' | 'error';
  person?: string;
  confidence?: number;
  best_match?: {
    identity: string;
    distance: number;
    confidence_metrics: {
      confidence_percentage: number;
      quality_score: string;
    };
  };
  recognized_persons?: Array<{
    person: string;
    confidence: number;
    match_count: number;
  }>;
  api_info?: {
    request_processing_time: number;
    total_faces_detected: number;
    valid_faces_after_filtering: number;
  };
}
```

---

## Workflow & UI Requirements

### User Journey

```
┌─────────────────────────────────────────────────────────────┐
│                     TRAINING UI WORKFLOW                     │
└─────────────────────────────────────────────────────────────┘

1. LOGIN
   ┌──────────────────┐
   │ Email Input      │
   │ [Get Token]      │
   └──────────────────┘
           ↓
2. GENERATE NAMES
   ┌──────────────────┐
   │ Country: [Serbia]│
   │ [Generate Names] │
   │                  │
   │ Status: 50 names │
   │ generated for    │
   │ 5 occupations    │
   └──────────────────┘
           ↓
3. PROCESS QUEUE
   ┌──────────────────┐
   │ Queue: 45/50     │
   │                  │
   │ Current:         │
   │ Dragan Bjelogrlic│
   │ [●●●●●●○○○○] 60% │
   │                  │
   │ [Process Next]   │
   │ [Process All]    │
   └──────────────────┘
           ↓
4. MONITOR PROGRESS
   ┌──────────────────┐
   │ Training Folders │
   │                  │
   │ Dragan_Bjelogrlic│
   │ ✓ 38 images      │
   │                  │
   │ Nebojsa_Glogovac │
   │ ⟳ 12 images      │
   │                  │
   │ Milos_Bikovic    │
   │ ✗ 3 images (low) │
   └──────────────────┘
           ↓
5. SYNC TO PRODUCTION
   ┌──────────────────┐
   │ Ready: 45 people │
   │                  │
   │ [Sync to Prod]   │
   └──────────────────┘
           ↓
6. TEST RECOGNITION
   ┌──────────────────┐
   │ Upload Test Image│
   │ [Choose File]    │
   │                  │
   │ Result:          │
   │ Dragan Bjelogrlic│
   │ Confidence: 95.2%│
   └──────────────────┘
```

### Required UI Components

#### 1. Login/Authentication Page
- Email input field
- "Get Token" button
- Display token (with copy button)
- Store token in localStorage or sessionStorage

#### 2. Dashboard (Overview)
- Current training statistics
  - Total people in queue
  - Processed count
  - In-progress count
  - Failed count
- Quick actions
  - Generate names button
  - Process next button
  - Sync to production button
- Recent activity log

#### 3. Name Generation Page
- Country selector (dropdown or input)
- "Generate Names" button
- Loading spinner during generation
- Success/error messages
- Display generated count

#### 4. Queue Management Page
- Table of people in queue
  - Name
  - Occupation
  - Status (pending/processing/completed)
  - Image count
  - Actions (process, skip, delete)
- Bulk actions
  - "Process All" button
  - "Process Next 10" button
- Filters
  - By occupation
  - By status
- Search functionality

#### 5. Training Progress Page
- Grid/list of training folders
  - Person name
  - Image count
  - Status indicator (ready/processing/insufficient)
  - Preview thumbnails
  - Actions (view details, delete, re-process)
- Filters
  - Show only ready (>5 images)
  - Show only insufficient (<5 images)
  - Show processing
- Refresh button (poll for updates)

#### 6. Image Browser
- Display all images for a person
- Grid view with thumbnails
- Actions per image
  - Delete
  - Mark as reference
  - View full size
- Bulk actions
  - Select multiple
  - Delete selected

#### 7. Sync to Production Page
- Source/target directory selection
- Preview of what will be synced
  - Number of people
  - Number of images
  - Total size
- "Start Sync" button
- Progress indicator
- Success/error messages

#### 8. Testing Page
- Two tabs:
  - **Single Recognition**: Test current system
  - **A/B Testing**: Compare current vs improved
- Image upload area (drag & drop or browse)
- Optional: ground truth input
- Display results
  - Person name
  - Confidence score
  - Processing time
  - Match details
- For A/B testing: side-by-side comparison

#### 9. Settings/Admin Page
- View name mappings
- API configuration
- System health status
- Logs viewer (if available)

---

## Example Code

### Authentication Setup

```javascript
// auth.js - Authentication service
class AuthService {
  constructor() {
    this.baseURL = 'http://localhost:5000';
    this.tokenKey = 'face_recognition_token';
  }

  async getToken(email) {
    try {
      const response = await fetch(`${this.baseURL}/api/auth/token-by-email`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ email })
      });

      const data = await response.json();

      if (data.success) {
        // Handle single or multiple tokens
        if (Array.isArray(data.data)) {
          // Multiple domains - let user choose or store all
          return data.data;
        } else {
          // Single domain
          this.saveToken(data.data.token);
          return data.data;
        }
      } else {
        throw new Error(data.error);
      }
    } catch (error) {
      console.error('Authentication error:', error);
      throw error;
    }
  }

  saveToken(token) {
    localStorage.setItem(this.tokenKey, token);
  }

  getStoredToken() {
    return localStorage.getItem(this.tokenKey);
  }

  clearToken() {
    localStorage.removeItem(this.tokenKey);
  }

  isAuthenticated() {
    return !!this.getStoredToken();
  }
}

export const authService = new AuthService();
```

### API Client Setup

```javascript
// api.js - API client with authentication
import { authService } from './auth';

class ApiClient {
  constructor() {
    this.baseURL = 'http://localhost:5000';
  }

  getHeaders(includeContentType = true) {
    const headers = {
      'Authorization': `Bearer ${authService.getStoredToken()}`
    };

    if (includeContentType) {
      headers['Content-Type'] = 'application/json';
    }

    return headers;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;

    const config = {
      ...options,
      headers: {
        ...this.getHeaders(options.includeContentType !== false),
        ...options.headers
      }
    };

    try {
      const response = await fetch(url, config);

      // Handle 401 Unauthorized
      if (response.status === 401) {
        authService.clearToken();
        throw new Error('Unauthorized - please login again');
      }

      const data = await response.json();
      return { data, status: response.status };
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // GET request
  async get(endpoint, params = {}) {
    const queryString = new URLSearchParams(params).toString();
    const url = queryString ? `${endpoint}?${queryString}` : endpoint;
    return this.request(url, { method: 'GET' });
  }

  // POST request
  async post(endpoint, body) {
    return this.request(endpoint, {
      method: 'POST',
      body: JSON.stringify(body)
    });
  }

  // POST with FormData (for file uploads)
  async postFormData(endpoint, formData) {
    return this.request(endpoint, {
      method: 'POST',
      body: formData,
      includeContentType: false // Let browser set Content-Type for FormData
    });
  }
}

export const api = new ApiClient();
```

### Training Operations

```javascript
// training.js - Training API methods
import { api } from './api';

export const trainingAPI = {
  // Generate celebrity names
  async generateNames(country) {
    const { data } = await api.get('/api/excel/check-excel', { country });
    return data;
  },

  // Process next person in queue
  async processNext() {
    const { data } = await api.get('/api/excel/process');
    return data;
  },

  // Upload manual image
  async uploadImage(imageFile, person, createdDate) {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('person', person);
    formData.append('created_date', createdDate);

    const { data } = await api.postFormData('/upload-with-domain', formData);
    return data;
  },

  // Test recognition
  async testRecognition(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);

    const { data } = await api.postFormData('/recognize', formData);
    return data;
  },

  // A/B test recognition
  async abTestRecognition(imageFile, imageId = null, groundTruth = null) {
    const formData = new FormData();
    formData.append('image', imageFile);
    if (imageId) formData.append('image_id', imageId);
    if (groundTruth) formData.append('ground_truth', groundTruth);

    const { data } = await api.postFormData('/api/test/recognize', formData);
    return data;
  },

  // Sync to production
  async syncToProduction(sourceDir = null, targetDir = null) {
    const params = {};
    if (sourceDir) params.source_dir = sourceDir;
    if (targetDir) params.target_dir = targetDir;

    const { data } = await api.get('/sync-faces', params);
    return data;
  },

  // Get name mappings
  async getNameMappings() {
    const { data } = await api.get('/admin/name-mappings');
    return data;
  },

  // Delete image
  async deleteImage(filename) {
    const { data } = await api.post('/manage-image', {
      filename,
      action: 'delete'
    });
    return data;
  }
};
```

### React Component Example - Name Generation

```jsx
// GenerateNamesPage.jsx
import React, { useState } from 'react';
import { trainingAPI } from './api/training';

function GenerateNamesPage() {
  const [country, setCountry] = useState('Serbia');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await trainingAPI.generateNames(country);

      if (data.success) {
        setResult(data);
      } else {
        setError(data.message || 'Failed to generate names');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="generate-names-page">
      <h1>Generate Celebrity Names</h1>

      <div className="form-group">
        <label>Country:</label>
        <input
          type="text"
          value={country}
          onChange={(e) => setCountry(e.target.value)}
          placeholder="Enter country name"
        />
      </div>

      <button
        onClick={handleGenerate}
        disabled={loading || !country}
      >
        {loading ? 'Generating...' : 'Generate Names'}
      </button>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && result.success && (
        <div className="success-message">
          <h3>Success!</h3>
          <p>{result.message}</p>
          <ul>
            <li>Rows in queue: {result.row_count}</li>
            <li>Country: {result.country}</li>
            <li>Processing: {result.thread_started ? 'Yes' : 'No'}</li>
          </ul>
        </div>
      )}
    </div>
  );
}

export default GenerateNamesPage;
```

### React Component Example - Process Queue

```jsx
// ProcessQueuePage.jsx
import React, { useState, useEffect } from 'react';
import { trainingAPI } from './api/training';

function ProcessQueuePage() {
  const [processing, setProcessing] = useState(false);
  const [currentResult, setCurrentResult] = useState(null);
  const [processedCount, setProcessedCount] = useState(0);
  const [error, setError] = useState(null);
  const [autoProcess, setAutoProcess] = useState(false);

  const processNext = async () => {
    setProcessing(true);
    setError(null);

    try {
      const data = await trainingAPI.processNext();

      if (data.success) {
        setCurrentResult(data);
        setProcessedCount(prev => prev + 1);
        return true; // Success
      } else {
        setError(data.message || 'Failed to process person');
        return false; // Failed
      }
    } catch (err) {
      setError(err.message);
      return false;
    } finally {
      setProcessing(false);
    }
  };

  const processAll = async () => {
    setAutoProcess(true);
    let success = true;

    while (success && autoProcess) {
      success = await processNext();
      // Wait 2 seconds between requests
      if (success) {
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }

    setAutoProcess(false);
  };

  const stopAutoProcess = () => {
    setAutoProcess(false);
  };

  return (
    <div className="process-queue-page">
      <h1>Process Training Queue</h1>

      <div className="stats">
        <p>Processed in this session: {processedCount}</p>
      </div>

      <div className="actions">
        <button
          onClick={processNext}
          disabled={processing || autoProcess}
        >
          {processing ? 'Processing...' : 'Process Next Person'}
        </button>

        {!autoProcess ? (
          <button onClick={processAll} disabled={processing}>
            Process All (Auto)
          </button>
        ) : (
          <button onClick={stopAutoProcess}>
            Stop Auto Processing
          </button>
        )}
      </div>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {currentResult && currentResult.success && (
        <div className="current-result">
          <h3>Currently Processing:</h3>
          <div className="person-info">
            <p><strong>Name:</strong> {currentResult.data.name} {currentResult.data.last_name}</p>
            <p><strong>Occupation:</strong> {currentResult.data.occupation}</p>
            {currentResult.data.original_name && (
              <p><strong>Original:</strong> {currentResult.data.original_name} {currentResult.data.original_last_name}</p>
            )}
          </div>

          {currentResult.images && (
            <div className="image-stats">
              <h4>Image Processing:</h4>
              <p>✓ Downloaded: {currentResult.images.count} images</p>
              <p>⟳ Total found: {currentResult.images.total_found}</p>
              <p>× Failed: {currentResult.images.failed.length}</p>
              <p className="note">
                Note: Background validation is running.
                This will take 2-5 minutes to verify all images.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default ProcessQueuePage;
```

### React Component Example - Test Recognition

```jsx
// TestRecognitionPage.jsx
import React, { useState } from 'react';
import { trainingAPI } from './api/training';

function TestRecognitionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState('single'); // 'single' or 'ab'

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  };

  const handleTest = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    try {
      let data;
      if (mode === 'single') {
        data = await trainingAPI.testRecognition(selectedFile);
      } else {
        data = await trainingAPI.abTestRecognition(selectedFile);
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="test-recognition-page">
      <h1>Test Face Recognition</h1>

      <div className="mode-selector">
        <button
          className={mode === 'single' ? 'active' : ''}
          onClick={() => setMode('single')}
        >
          Single Test
        </button>
        <button
          className={mode === 'ab' ? 'active' : ''}
          onClick={() => setMode('ab')}
        >
          A/B Test (Compare Systems)
        </button>
      </div>

      <div className="upload-area">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
        />
        {preview && (
          <img src={preview} alt="Preview" style={{ maxWidth: '300px' }} />
        )}
      </div>

      <button
        onClick={handleTest}
        disabled={!selectedFile || loading}
      >
        {loading ? 'Testing...' : 'Test Recognition'}
      </button>

      {error && (
        <div className="error-message">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && mode === 'single' && (
        <div className="result single-result">
          <h3>Recognition Result:</h3>
          {result.status === 'success' ? (
            <>
              <p className="person-name">
                <strong>Recognized as:</strong> {result.person}
              </p>
              <p><strong>Confidence:</strong> {result.confidence}%</p>
              <p><strong>Processing Time:</strong> {result.api_info?.request_processing_time}s</p>

              {result.best_match && (
                <div className="match-details">
                  <h4>Best Match:</h4>
                  <p>File: {result.best_match.identity}</p>
                  <p>Distance: {result.best_match.distance}</p>
                  <p>Quality: {result.best_match.confidence_metrics?.quality_score}</p>
                </div>
              )}
            </>
          ) : (
            <p className="no-match">No face recognized</p>
          )}
        </div>
      )}

      {result && mode === 'ab' && (
        <div className="result ab-result">
          <h3>A/B Test Results:</h3>

          <div className="comparison">
            <div className="pipeline">
              <h4>Pipeline A (Current - VGG-Face)</h4>
              <p><strong>Status:</strong> {result.pipeline_a_result.status}</p>
              <p><strong>Person:</strong> {result.pipeline_a_result.person || 'None'}</p>
              <p><strong>Confidence:</strong> {result.pipeline_a_result.confidence}%</p>
              <p><strong>Time:</strong> {result.pipeline_a_result.processing_time}s</p>
            </div>

            <div className="pipeline">
              <h4>Pipeline B (Improved - Facenet512)</h4>
              <p><strong>Status:</strong> {result.pipeline_b_result.status}</p>
              <p><strong>Person:</strong> {result.pipeline_b_result.person || 'None'}</p>
              <p><strong>Confidence:</strong> {result.pipeline_b_result.confidence}%</p>
              <p><strong>Time:</strong> {result.pipeline_b_result.processing_time}s</p>
            </div>
          </div>

          <div className="recommendation">
            <h4>Analysis:</h4>
            <p>{result.recommendation}</p>

            {result.comparison.comparison_metrics.both_succeeded && (
              <>
                <p>
                  <strong>Agreement:</strong>{' '}
                  {result.comparison.comparison_metrics.results_match ? '✓ Match' : '✗ Disagree'}
                </p>
                <p>
                  <strong>Confidence Difference:</strong>{' '}
                  {result.comparison.comparison_metrics.confidence_difference > 0 ? '+' : ''}
                  {result.comparison.comparison_metrics.confidence_difference}%
                </p>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default TestRecognitionPage;
```

---

## Error Handling

### Common HTTP Status Codes

| Code | Meaning | Action |
|------|---------|--------|
| 200 | Success | Parse and display data |
| 202 | Accepted (async) | Show "Processing started" message |
| 400 | Bad Request | Show validation error to user |
| 401 | Unauthorized | Clear token, redirect to login |
| 404 | Not Found | Show "Resource not found" error |
| 500 | Server Error | Show "Server error, try again" message |

### Error Response Format

All errors follow this format:
```json
{
  "success": false,
  "error": "Error message here",
  "message": "Alternative error field"
}
```

### Error Handling Best Practices

```javascript
async function handleApiCall(apiFunction) {
  try {
    const result = await apiFunction();

    // Check success flag
    if (result.success === false) {
      throw new Error(result.error || result.message);
    }

    return result;
  } catch (error) {
    // Network error
    if (error.message === 'Failed to fetch') {
      return {
        success: false,
        error: 'Network error. Please check your connection.'
      };
    }

    // Auth error
    if (error.message.includes('Unauthorized')) {
      // Clear token and redirect to login
      authService.clearToken();
      window.location.href = '/login';
      return;
    }

    // Generic error
    return {
      success: false,
      error: error.message || 'An unexpected error occurred'
    };
  }
}
```

---

## Testing

### Manual Testing Checklist

#### Authentication
- [ ] Can login with valid email
- [ ] Shows error with invalid email
- [ ] Token is stored in localStorage
- [ ] Token is sent in Authorization header
- [ ] Redirect to login on 401

#### Name Generation
- [ ] Can generate names for a country
- [ ] Shows loading state during generation
- [ ] Displays success message with count
- [ ] Handles errors gracefully

#### Queue Processing
- [ ] Can process single person
- [ ] Shows current person details
- [ ] Displays image download statistics
- [ ] Can process multiple in sequence
- [ ] Can stop auto-processing

#### Testing
- [ ] Can upload image for recognition
- [ ] Shows preview of uploaded image
- [ ] Displays recognition result
- [ ] Shows confidence and details
- [ ] A/B test shows both results side-by-side

#### Error Handling
- [ ] Network errors are caught and displayed
- [ ] Invalid tokens trigger re-authentication
- [ ] Server errors show helpful messages
- [ ] Form validation prevents empty submissions

### Example Test Data

**Test Emails** (ask backend team for valid emails):
```
test@example.com
admin@example.com
```

**Test Countries**:
```
Serbia
United States
France
United Kingdom
```

**Test Images**:
- Download celebrity images from Google
- Use test images with clear, frontal faces
- Test with multiple faces (should fail)
- Test with blurry images (should fail)
- Test with no faces (should fail)

---

## Additional Resources

### Backend Architecture
See `TRAINING_DATA_COLLECTION_GUIDE.md` for complete system architecture

### API Testing Tools
- **Postman**: Import collection with all endpoints
- **cURL**: Command-line testing examples provided above
- **Browser DevTools**: Network tab for debugging

### Recommended Libraries

**React**:
```json
{
  "axios": "^1.6.0",
  "react-query": "^3.39.0",
  "react-dropzone": "^14.2.3",
  "react-table": "^7.8.0"
}
```

**Vue**:
```json
{
  "axios": "^1.6.0",
  "@vueuse/core": "^10.0.0",
  "vue-query": "^1.26.0"
}
```

### Environment Variables

Create `.env` file in frontend:
```env
VITE_API_BASE_URL=http://localhost:5000
VITE_API_TIMEOUT=30000
```

---

## FAQ

**Q: How often should I poll for status updates?**
A: For background processes, poll every 5-10 seconds. For queue status, every 30 seconds is sufficient.

**Q: What image formats are supported?**
A: JPG, JPEG, PNG, GIF, BMP, WebP

**Q: What's the maximum file size for uploads?**
A: 30MB (configured in Flask: `MAX_CONTENT_LENGTH = 30 * 1024 * 1024`)

**Q: How long does processing take?**
A:
- Name generation: 30-60 seconds
- Image download: 5-15 seconds (API response)
- Background validation: 2-5 minutes
- Recognition test: 2-4 seconds

**Q: Can I process multiple people simultaneously?**
A: Yes, but be mindful of API rate limits and server resources. Processing 2-3 in parallel is safe.

**Q: What happens if processing fails?**
A: The person remains in the queue and can be retried. Failed images are logged.

**Q: How do I know when background processing is complete?**
A: Poll the training folder status endpoint (to be implemented) or check the filesystem manually.

---

## Support & Contact

For questions or issues:
- Check backend logs for detailed error messages
- Review `TRAINING_DATA_COLLECTION_GUIDE.md` for system architecture
- Test endpoints with Postman/cURL first
- Verify authentication token is valid

---

## Changelog

**v1.0 - 2025-01-13**
- Initial documentation
- Core training endpoints
- A/B testing endpoints
- Authentication flow
- Example code for React
