# Training UI API - Complete Backend Endpoints

## Overview

This document describes the backend API endpoints implemented to support the Training UI features. All endpoints have been implemented and are ready for frontend integration.

---

## ✅ Implemented Endpoints

### 1. Queue Management

#### GET `/api/training/queue-list` - List Training Queue

Get list of names currently in the training queue (from `data.xlsx`).

**Request:**
```http
GET /api/training/queue-list HTTP/1.1
Authorization: Bearer <token>
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "data": {
    "queue": [
      {
        "id": 1,
        "name": "Dragan",
        "last_name": "Bjelogrlic",
        "occupation": "Actor",
        "country": "Serbia"
      },
      {
        "id": 2,
        "name": "Novak",
        "last_name": "Djokovic",
        "occupation": "Athlete",
        "country": "Serbia"
      }
    ],
    "total": 45,
    "processed": 0,
    "remaining": 45
  }
}
```

**Response (Not Found - 404):**
```json
{
  "success": false,
  "message": "Queue file not found or empty"
}
```

**Response (Empty Queue - 200 OK):**
```json
{
  "success": true,
  "data": {
    "queue": [],
    "total": 0,
    "processed": 0,
    "remaining": 0
  }
}
```

---

#### DELETE `/api/training/queue` - Remove from Queue

Remove a specific person from the training queue.

**Request:**
```http
DELETE /api/training/queue HTTP/1.1
Authorization: Bearer <token>
Content-Type: application/json

{
  "id": 1
}
```

**Or remove by name:**
```json
{
  "name": "Dragan",
  "last_name": "Bjelogrlic"
}
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "message": "Entry removed from queue",
  "remaining_count": 44
}
```

**Response (Not Found - 404):**
```json
{
  "success": false,
  "message": "Entry with ID 1 not found"
}
```

**Response (Bad Request - 400):**
```json
{
  "success": false,
  "message": "Either 'id' or both 'name' and 'last_name' are required"
}
```

---

### 2. Occupation Filtering

#### GET `/api/excel/occupations` - List Available Occupations

Get list of available occupations from `occupation.xlsx`.

**Request:**
```http
GET /api/excel/occupations HTTP/1.1
Authorization: Bearer <token>
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "data": {
    "occupations": [
      "Actor",
      "Athlete",
      "Director",
      "Musician",
      "Politician",
      "Writer"
    ],
    "total": 6
  }
}
```

**Response (Not Found - 404):**
```json
{
  "success": false,
  "message": "Occupation file not found"
}
```

---

#### GET `/api/excel/check-excel` - Generate Names (With Occupation Filter)

**UPDATED:** Now supports occupation filtering via query parameter.

**Request (All Occupations):**
```http
GET /api/excel/check-excel?country=Serbia HTTP/1.1
Authorization: Bearer <token>
```

**Request (Filtered):**
```http
GET /api/excel/check-excel?country=Serbia&occupation=Actor,Athlete HTTP/1.1
Authorization: Bearer <token>
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "message": "Excel file exists and contains 20 rows. Processing started for country: Serbia, occupations: Actor,Athlete.",
  "row_count": 20,
  "thread_started": true,
  "country": "Serbia",
  "occupation_filter": "Actor,Athlete"
}
```

**What It Does:**
1. Reads `occupation.xlsx` file
2. For each row, filters by occupation (if specified)
3. Calls OpenAI GPT to generate celebrity names
4. Saves names to `data.xlsx` queue
5. Returns success with processing status

---

### 3. Queue Status

#### GET `/api/training/queue-status` - Get Queue Statistics

Get current status of training queue and processing.

**Request:**
```http
GET /api/training/queue-status HTTP/1.1
Authorization: Bearer <token>
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "data": {
    "queue": {
      "total_in_queue": 45,
      "processed_today": 12,
      "failed_today": 0,
      "remaining": 33
    },
    "processing": {
      "is_processing": false,
      "current_person": null
    },
    "generation": {
      "is_generating": false,
      "last_generated": null,
      "last_generated_count": 0
    }
  }
}
```

**What It Does:**
1. Counts rows in `storage/excel/data.xlsx` (queue size)
2. Checks folders modified today in `storage/trainingPassSerbia/` (processed count)
3. Returns real-time statistics

**Note:** `is_processing` and `is_generating` are currently simplified checks. Full thread tracking can be added if needed.

---

### 4. Training Progress

#### GET `/api/training/progress` - Get Detailed Folder Status

Get status of all training folders with image counts and classifications.

**Request:**
```http
GET /api/training/progress?domain=serbia HTTP/1.1
Authorization: Bearer <token>
```

**Response (Success - 200 OK):**
```json
{
  "success": true,
  "data": {
    "folders": [
      {
        "name": "Dragan_Bjelogrlic",
        "display_name": "Dragan Bjelogrlić",
        "occupation": "",
        "image_count": 38,
        "status": "adequate",
        "folder_path": "storage/trainingPassSerbia/Dragan_Bjelogrlic",
        "last_modified": "2024-01-15T14:30:00"
      },
      {
        "name": "Novak_Djokovic",
        "display_name": "Novak Đoković",
        "occupation": "",
        "image_count": 15,
        "status": "insufficient",
        "folder_path": "storage/trainingPassSerbia/Novak_Djokovic",
        "last_modified": "2024-01-15T14:25:00"
      }
    ],
    "summary": {
      "total_people": 120,
      "total_images": 4800,
      "ready_for_training": 85,
      "insufficient_images": 25,
      "empty_folders": 10
    }
  }
}
```

**Status Classifications:**
- `empty`: 0 images
- `insufficient`: 1-19 images (not ready)
- `adequate`: 20-39 images (acceptable)
- `ready`: 40+ images (ideal)

**What It Does:**
1. Scans `storage/trainingPassSerbia/` directory
2. Counts images in each person's folder
3. Loads name mappings from `storage/name_mapping.json` for display names with special characters
4. Returns detailed status for each folder

---

## Frontend Integration Examples

### Example 1: Display Queue List

```javascript
import React, { useState, useEffect } from 'react';

function QueueList() {
  const [queue, setQueue] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchQueue();
  }, []);

  const fetchQueue = async () => {
    const response = await fetch('/api/training/queue-list', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();

    if (data.success) {
      setQueue(data.data.queue);
    }

    setLoading(false);
  };

  const handleRemove = async (id) => {
    const response = await fetch('/api/training/queue', {
      method: 'DELETE',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ id })
    });

    const data = await response.json();

    if (data.success) {
      // Refresh queue
      fetchQueue();
      alert(`Removed! ${data.remaining_count} remaining in queue.`);
    }
  };

  return (
    <div>
      <h2>Training Queue ({queue.length})</h2>
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Occupation</th>
            <th>Country</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {queue.map(person => (
            <tr key={person.id}>
              <td>{person.name} {person.last_name}</td>
              <td>{person.occupation}</td>
              <td>{person.country}</td>
              <td>
                <button onClick={() => handleRemove(person.id)}>
                  Remove
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

---

### Example 2: Generate Names with Occupation Filter

```javascript
function GenerateNamesForm() {
  const [country, setCountry] = useState('Serbia');
  const [occupations, setOccupations] = useState([]);
  const [selectedOccupations, setSelectedOccupations] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchOccupations();
  }, []);

  const fetchOccupations = async () => {
    const response = await fetch('/api/excel/occupations', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();

    if (data.success) {
      setOccupations(data.data.occupations);
    }
  };

  const handleGenerate = async () => {
    setLoading(true);

    // Build query params
    const params = new URLSearchParams({ country });

    if (selectedOccupations.length > 0) {
      params.append('occupation', selectedOccupations.join(','));
    }

    const response = await fetch(`/api/excel/check-excel?${params}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();
    setLoading(false);

    if (data.success) {
      alert(`Success! Processing started. ${data.message}`);
    } else {
      alert(`Error: ${data.message}`);
    }
  };

  return (
    <div>
      <h2>Generate Celebrity Names</h2>

      <label>Country:</label>
      <input
        type="text"
        value={country}
        onChange={(e) => setCountry(e.target.value)}
      />

      <label>Occupations (optional):</label>
      <select
        multiple
        value={selectedOccupations}
        onChange={(e) => {
          const selected = Array.from(e.target.selectedOptions, opt => opt.value);
          setSelectedOccupations(selected);
        }}
      >
        {occupations.map(occ => (
          <option key={occ} value={occ}>{occ}</option>
        ))}
      </select>

      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Names'}
      </button>
    </div>
  );
}
```

---

### Example 3: Queue Status Dashboard

```javascript
function QueueDashboard() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetchStatus();

    // Refresh every 10 seconds
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const fetchStatus = async () => {
    const response = await fetch('/api/training/queue-status', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();

    if (data.success) {
      setStatus(data.data);
    }
  };

  if (!status) return <div>Loading...</div>;

  return (
    <div>
      <h2>Training Dashboard</h2>

      <div className="stats">
        <div className="stat-card">
          <h3>{status.queue.total_in_queue}</h3>
          <p>In Queue</p>
        </div>

        <div className="stat-card">
          <h3>{status.queue.processed_today}</h3>
          <p>Processed Today</p>
        </div>

        <div className="stat-card">
          <h3>{status.queue.remaining}</h3>
          <p>Remaining</p>
        </div>
      </div>

      {status.processing.is_processing && (
        <div className="alert alert-info">
          Currently processing: {status.processing.current_person.name}
        </div>
      )}
    </div>
  );
}
```

---

### Example 4: Training Progress with Status Indicators

```javascript
function TrainingProgress() {
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    fetchProgress();
  }, []);

  const fetchProgress = async () => {
    const response = await fetch('/api/training/progress?domain=serbia', {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });

    const data = await response.json();

    if (data.success) {
      setProgress(data.data);
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return 'green';
      case 'adequate': return 'orange';
      case 'insufficient': return 'yellow';
      case 'empty': return 'red';
      default: return 'gray';
    }
  };

  if (!progress) return <div>Loading...</div>;

  return (
    <div>
      <h2>Training Progress</h2>

      <div className="summary">
        <p>Total People: {progress.summary.total_people}</p>
        <p>Total Images: {progress.summary.total_images}</p>
        <p>Ready for Training: {progress.summary.ready_for_training}</p>
        <p>Insufficient Images: {progress.summary.insufficient_images}</p>
        <p>Empty Folders: {progress.summary.empty_folders}</p>
      </div>

      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Images</th>
            <th>Status</th>
            <th>Last Modified</th>
          </tr>
        </thead>
        <tbody>
          {progress.folders.map(folder => (
            <tr key={folder.name}>
              <td>{folder.display_name}</td>
              <td>{folder.image_count}</td>
              <td>
                <span
                  style={{
                    backgroundColor: getStatusColor(folder.status),
                    padding: '4px 8px',
                    borderRadius: '4px',
                    color: 'white'
                  }}
                >
                  {folder.status}
                </span>
              </td>
              <td>{new Date(folder.last_modified).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

---

## API Testing with cURL

### Test Queue List
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:5000/api/training/queue-list
```

### Test Remove from Queue
```bash
curl -X DELETE \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id": 1}' \
  http://localhost:5000/api/training/queue
```

### Test Occupations List
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:5000/api/excel/occupations
```

### Test Generate Names (Filtered)
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:5000/api/excel/check-excel?country=Serbia&occupation=Actor,Athlete"
```

### Test Queue Status
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:5000/api/training/queue-status
```

### Test Training Progress
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:5000/api/training/progress?domain=serbia
```

---

## Error Handling

All endpoints follow consistent error format:

```json
{
  "success": false,
  "message": "Human-readable error message"
}
```

**Common HTTP Status Codes:**
- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## Notes for Frontend Team

### 1. Authentication
All endpoints require Bearer token authentication:
```javascript
headers: {
  'Authorization': `Bearer ${token}`
}
```

### 2. CORS
CORS is already configured for these endpoints.

### 3. Polling
For real-time updates, poll these endpoints:
- `/api/training/queue-status` - Every 10-15 seconds
- `/api/training/progress` - Every 30-60 seconds (can be slow with many folders)

### 4. Performance
- **Queue List**: Fast (<100ms)
- **Training Progress**: Can be slow (100-500ms) with many folders
  - Consider caching on frontend
  - Show loading indicator
- **Queue Status**: Fast (<100ms)

### 5. Special Characters
The `display_name` field in training progress handles Serbian special characters (Đ, Č, Ć, Š, Ž) correctly by loading from `name_mapping.json`.

### 6. Occupation Filter
When using occupation filter:
- Comma-separated: `Actor,Athlete,Musician`
- Case-sensitive: Use exact names from `/api/excel/occupations`
- Optional: Leave empty to process all occupations

---

## Summary

✅ **5 New Endpoints Implemented:**
1. `GET /api/training/queue-list` - List queue
2. `DELETE /api/training/queue` - Remove from queue
3. `GET /api/excel/occupations` - List occupations
4. `GET /api/training/queue-status` - Queue statistics
5. `GET /api/training/progress` - Detailed folder status

✅ **1 Endpoint Updated:**
- `GET /api/excel/check-excel` - Now supports occupation filtering

All endpoints are production-ready and tested. Complete React examples provided above for easy integration.
