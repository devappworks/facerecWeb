# Wikidata-Based Celebrity Training Workflow

## Overview

This document describes the **NEW** celebrity training workflow using Wikidata instead of OpenAI GPT. This approach is:
- ✅ **FREE** - No API costs
- ✅ **Reliable** - Real people from Wikipedia's structured database
- ✅ **Comprehensive** - Hundreds of celebrities per country/occupation
- ✅ **Smart** - Automatically checks if person already exists in database

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Training Workflow                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Generate Candidates (Wikidata)                          │
│     ↓                                                        │
│  2. Check Existing Database                                 │
│     ↓                                                        │
│  3. User Selects People to Train                            │
│     ↓                                                        │
│  4. Batch Download & Validate (SERP + DeepFace)            │
│     ↓                                                        │
│  5. Review Results                                           │
│     ↓                                                        │
│  6. Deploy to Production                                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## API Endpoints

### 1. Get Available Countries

**GET** `/api/training/countries`

Returns list of supported countries.

**Response:**
```json
{
  "success": true,
  "countries": [
    {"id": "serbia", "name": "Serbia"},
    {"id": "usa", "name": "Usa"},
    {"id": "uk", "name": "Uk"},
    {"id": "france", "name": "France"},
    {"id": "croatia", "name": "Croatia"},
    {"id": "bosnia", "name": "Bosnia"},
    ...
  ]
}
```

---

### 2. Get Available Occupations

**GET** `/api/training/occupations`

Returns list of supported occupations.

**Response:**
```json
{
  "success": true,
  "occupations": [
    {"id": "actor", "name": "Actor"},
    {"id": "politician", "name": "Politician"},
    {"id": "tennis_player", "name": "Tennis Player"},
    {"id": "football_player", "name": "Football Player"},
    {"id": "basketball_player", "name": "Basketball Player"},
    {"id": "musician", "name": "Musician"},
    {"id": "singer", "name": "Singer"},
    {"id": "writer", "name": "Writer"},
    {"id": "director", "name": "Director"},
    ...
  ]
}
```

---

### 3. Generate Candidates from Wikidata

**POST** `/api/training/generate-candidates`

Query Wikidata for celebrities and check against existing database.

**Request Body:**
```json
{
  "country": "serbia",
  "occupation": "actor",
  "domain": "serbia"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Found 287 serbia actors",
  "candidates": [
    {
      "full_name": "Novak Djokovic",
      "name": "Novak",
      "last_name": "Djokovic",
      "occupation": "tennis_player",
      "country": "serbia",
      "description": "Serbian tennis player",
      "wikidata_id": "Q5812",
      "has_wikipedia_image": true,
      "exists_in_db": false,
      "existing_photo_count": 0,
      "folder_name": "novak_djokovic"
    },
    {
      "full_name": "Emir Kusturica",
      "name": "Emir",
      "last_name": "Kusturica",
      "occupation": "director",
      "country": "serbia",
      "description": "Serbian film director",
      "wikidata_id": "Q55411",
      "has_wikipedia_image": true,
      "exists_in_db": true,
      "existing_photo_count": 47,
      "folder_name": "emir_kusturica"
    }
  ],
  "statistics": {
    "total": 287,
    "new": 215,
    "existing": 72
  }
}
```

**Key Fields:**
- `exists_in_db`: If `true`, this person is already trained
- `existing_photo_count`: Number of photos already in production database
- `has_wikipedia_image`: If `true`, person has an image on Wikipedia
- `folder_name`: Normalized folder name used for storage

---

### 4. Search for Specific Person

**GET** `/api/training/search-person?query=novak&limit=20`

Search Wikidata for a specific person (used for autocomplete).

**Query Parameters:**
- `query` (required): Search term (minimum 2 characters)
- `limit` (optional): Maximum results (default 20)

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "full_name": "Novak Djokovic",
      "name": "Novak",
      "last_name": "Djokovic",
      "description": "Serbian tennis player",
      "occupation": "Tennis Player",
      "country": "Serbia",
      "image_url": "https://upload.wikimedia.org/...",
      "wikidata_id": "Q5812"
    }
  ],
  "count": 1
}
```

---

### 5. Start Batch Training

**POST** `/api/training/start-batch`

Start downloading and processing images for selected celebrities.

**Request Body:**
```json
{
  "candidates": [
    {
      "full_name": "Novak Djokovic",
      "name": "Novak",
      "last_name": "Djokovic",
      "occupation": "tennis_player",
      "folder_name": "novak_djokovic"
    },
    {
      "full_name": "Ana Ivanovic",
      "name": "Ana",
      "last_name": "Ivanovic",
      "occupation": "tennis_player",
      "folder_name": "ana_ivanovic"
    }
  ],
  "domain": "serbia"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Started batch training for 2 people",
  "batch_id": "a1b2c3d4",
  "total_people": 2
}
```

**Important:** Save the `batch_id` to track progress!

---

### 6. Get Batch Training Status

**GET** `/api/training/batch/{batch_id}/status`

Poll this endpoint every 3 seconds to get real-time progress.

**Response:**
```json
{
  "success": true,
  "batch_id": "a1b2c3d4",
  "status": "processing",
  "created_at": "2025-11-14T10:30:00",
  "total": 2,
  "completed": 1,
  "processing": 1,
  "failed": 0,
  "queued": 0,
  "progress_percentage": 50,
  "people": [
    {
      "full_name": "Novak Djokovic",
      "name": "Novak",
      "last_name": "Djokovic",
      "occupation": "tennis_player",
      "folder_name": "novak_djokovic",
      "status": "completed",
      "current_step": "completed",
      "photos_downloaded": 28,
      "photos_validated": 0,
      "error": null,
      "started_at": "2025-11-14T10:30:05",
      "completed_at": "2025-11-14T10:32:18"
    },
    {
      "full_name": "Ana Ivanovic",
      "name": "Ana",
      "last_name": "Ivanovic",
      "occupation": "tennis_player",
      "folder_name": "ana_ivanovic",
      "status": "processing",
      "current_step": "downloading_images",
      "photos_downloaded": 15,
      "photos_validated": 0,
      "error": null,
      "started_at": "2025-11-14T10:32:20",
      "completed_at": null
    }
  ]
}
```

**Status Values:**
- `queued` - Waiting to start
- `processing` - Currently downloading/validating images
- `completed` - Finished successfully
- `failed` - Error occurred

**Current Step Values:**
- `downloading_images` - Downloading from Google via SERP API
- `validating_faces` - DeepFace is validating images
- `completed` - Done

**Batch Status:**
- `processing` - Still working
- `completed` - All people processed

---

### 7. Get Staging List

**GET** `/api/training/staging-list?domain=serbia`

Get list of people in staging area (trainingPassSerbia) ready for deployment.

**Response:**
```json
{
  "success": true,
  "people": [
    {
      "folder_name": "novak_djokovic",
      "image_count": 28,
      "ready_for_production": true
    },
    {
      "folder_name": "ana_ivanovic",
      "image_count": 31,
      "ready_for_production": true
    },
    {
      "folder_name": "unknown_person",
      "image_count": 3,
      "ready_for_production": false
    }
  ],
  "count": 3,
  "ready_count": 2
}
```

**Note:** `ready_for_production` is `true` if `image_count >= 5`.

---

### 8. Deploy to Production

**POST** `/api/training/deploy`

Move trained people from staging to production database.

**Request Body:**
```json
{
  "people": ["novak_djokovic", "ana_ivanovic"],
  "domain": "serbia"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Deployed 2 people to production",
  "deployed": [
    {
      "folder": "novak_djokovic",
      "image_count": 28
    },
    {
      "folder": "ana_ivanovic",
      "image_count": 31
    }
  ],
  "skipped": [
    {
      "folder": "unknown_person",
      "reason": "Too few images (3), minimum is 5"
    }
  ],
  "errors": [],
  "statistics": {
    "deployed_count": 2,
    "skipped_count": 1,
    "error_count": 0
  }
}
```

**What Happens:**
- Folders with 5+ images are moved to `storage/recognized_faces_prod/serbia/`
- Folders with <5 images are deleted automatically
- If folder exists in production, images are merged

---

## Complete UI Workflow

### Page 1: Generate Candidates

```
┌────────────────────────────────────────────────────────┐
│  Celebrity Training - Generate Candidates              │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Country: [Serbia ▼]    Occupation: [Actor ▼]         │
│                                                         │
│  [Generate from Wikidata]                              │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ Found 287 Serbian Actors                         │ │
│  │                                                   │ │
│  │ Filter: [Show only NEW] [Show ALL]               │ │
│  │                                                   │ │
│  │ ☑ Novak Djokovic          ✨ NEW                │ │
│  │ ☐ Emir Kusturica          ✅ EXISTS (47 photos) │ │
│  │ ☑ Milena Dravic           ✨ NEW                │ │
│  │ ☑ Rade Serbedzija         ✨ NEW                │ │
│  │ ☐ Predrag Manojlovic      ✅ EXISTS (32 photos) │ │
│  │ ...                                               │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  Selected: 215 new people                              │
│  [Start Training Process] [Cancel]                     │
└────────────────────────────────────────────────────────┘
```

**Implementation:**

```javascript
// 1. Load countries and occupations on page load
const loadOptions = async () => {
  const countriesResp = await fetch('/api/training/countries');
  const countries = await countriesResp.json();

  const occupationsResp = await fetch('/api/training/occupations');
  const occupations = await occupationsResp.json();

  // Populate dropdowns
};

// 2. Generate candidates
const generateCandidates = async () => {
  const response = await fetch('/api/training/generate-candidates', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      country: selectedCountry,
      occupation: selectedOccupation,
      domain: 'serbia'
    })
  });

  const data = await response.json();
  setCandidates(data.candidates);
  setStatistics(data.statistics);
};

// 3. Start training
const startTraining = async () => {
  const selected = candidates.filter(c => c.isSelected);

  const response = await fetch('/api/training/start-batch', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      candidates: selected,
      domain: 'serbia'
    })
  });

  const data = await response.json();
  const batchId = data.batch_id;

  // Navigate to progress page
  navigate(`/training/progress/${batchId}`);
};
```

---

### Page 2: Training Progress

```
┌────────────────────────────────────────────────────────┐
│  Training in Progress                                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Batch ID: a1b2c3d4                                    │
│  Started: 2025-11-14 10:30:00                          │
│                                                         │
│  Overall Progress: 85/215 completed (40%)              │
│  ████████████░░░░░░░░░░░░░░░ 40%                      │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ ✅ Novak Djokovic - COMPLETED (28 photos)       │ │
│  │ ✅ Ana Ivanovic - COMPLETED (31 photos)         │ │
│  │ ✅ Milena Dravic - COMPLETED (24 photos)        │ │
│  │ 🔄 Rade Serbedzija - PROCESSING (15/70 photos)  │ │
│  │ ⏳ Predrag Manojlovic - QUEUED                   │ │
│  │ ⏳ Nikola Jokic - QUEUED                         │ │
│  │ ❌ Unknown Person - FAILED (error message)       │ │
│  │ ...                                               │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  [View Logs] [Pause] [Cancel All]                     │
└────────────────────────────────────────────────────────┘
```

**Implementation:**

```javascript
// Poll every 3 seconds
useEffect(() => {
  const interval = setInterval(async () => {
    const response = await fetch(`/api/training/batch/${batchId}/status`);
    const data = await response.json();

    setBatchStatus(data);

    // If completed, stop polling and navigate
    if (data.status === 'completed') {
      clearInterval(interval);
      setTimeout(() => {
        navigate('/training/review');
      }, 2000);
    }
  }, 3000);

  return () => clearInterval(interval);
}, [batchId]);

// Calculate progress
const progressPercentage = batchStatus.progress_percentage;
const statusIcons = {
  'completed': '✅',
  'processing': '🔄',
  'queued': '⏳',
  'failed': '❌'
};
```

---

### Page 3: Review & Deploy

```
┌────────────────────────────────────────────────────────┐
│  Review Training Results                               │
├────────────────────────────────────────────────────────┤
│                                                         │
│  Training completed! Review results below:             │
│                                                         │
│  Ready for Production (198 people)                     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐ │
│  │ ☑ Novak Djokovic - 28 photos ✓                  │ │
│  │ ☑ Ana Ivanovic - 31 photos ✓                    │ │
│  │ ☑ Milena Dravic - 24 photos ✓                   │ │
│  │ ☐ Unknown Person - 3 photos ⚠️ (too few!)       │ │
│  │ ☐ Failed Person - 0 photos ❌ (validation error)│ │
│  │ ...                                               │ │
│  └──────────────────────────────────────────────────┘ │
│                                                         │
│  [Preview Photos] [Deploy 198 to Production]           │
└────────────────────────────────────────────────────────┘
```

**Implementation:**

```javascript
// 1. Load staging list
const loadStagingList = async () => {
  const response = await fetch('/api/training/staging-list?domain=serbia');
  const data = await response.json();

  setStagingPeople(data.people);
};

// 2. Deploy selected to production
const deployToProduction = async () => {
  const selected = stagingPeople
    .filter(p => p.isSelected && p.ready_for_production)
    .map(p => p.folder_name);

  const response = await fetch('/api/training/deploy', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      people: selected,
      domain: 'serbia'
    })
  });

  const data = await response.json();

  alert(`Successfully deployed ${data.deployed.length} people!`);

  // Show details
  console.log('Deployed:', data.deployed);
  console.log('Skipped:', data.skipped);
  console.log('Errors:', data.errors);
};
```

---

## React Component Examples

### Complete Training Flow Component

```jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Button, Select, MenuItem, Checkbox,
  List, ListItem, LinearProgress, Typography, Chip
} from '@mui/material';

function TrainingWorkflow() {
  const [step, setStep] = useState(1); // 1=Generate, 2=Progress, 3=Review
  const [countries, setCountries] = useState([]);
  const [occupations, setOccupations] = useState([]);
  const [selectedCountry, setSelectedCountry] = useState('serbia');
  const [selectedOccupation, setSelectedOccupation] = useState('actor');
  const [candidates, setCandidates] = useState([]);
  const [batchId, setBatchId] = useState(null);
  const [batchStatus, setBatchStatus] = useState(null);

  // Load options on mount
  useEffect(() => {
    loadOptions();
  }, []);

  const loadOptions = async () => {
    const [countriesRes, occupationsRes] = await Promise.all([
      fetch('/api/training/countries'),
      fetch('/api/training/occupations')
    ]);

    setCountries(await countriesRes.json());
    setOccupations(await occupationsRes.json());
  };

  const generateCandidates = async () => {
    const response = await fetch('/api/training/generate-candidates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        country: selectedCountry,
        occupation: selectedOccupation,
        domain: 'serbia'
      })
    });

    const data = await response.json();
    setCandidates(data.candidates.map(c => ({...c, isSelected: !c.exists_in_db})));
  };

  const startTraining = async () => {
    const selected = candidates.filter(c => c.isSelected);

    const response = await fetch('/api/training/start-batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        candidates: selected,
        domain: 'serbia'
      })
    });

    const data = await response.json();
    setBatchId(data.batch_id);
    setStep(2);
  };

  // Poll batch status
  useEffect(() => {
    if (!batchId) return;

    const interval = setInterval(async () => {
      const response = await fetch(`/api/training/batch/${batchId}/status`);
      const data = await response.json();
      setBatchStatus(data);

      if (data.status === 'completed') {
        clearInterval(interval);
        setStep(3);
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [batchId]);

  // Render based on current step
  if (step === 1) {
    return (
      <Box>
        <Typography variant="h4">Generate Training Candidates</Typography>

        <Select value={selectedCountry} onChange={e => setSelectedCountry(e.target.value)}>
          {countries.map(c => <MenuItem key={c.id} value={c.id}>{c.name}</MenuItem>)}
        </Select>

        <Select value={selectedOccupation} onChange={e => setSelectedOccupation(e.target.value)}>
          {occupations.map(o => <MenuItem key={o.id} value={o.id}>{o.name}</MenuItem>)}
        </Select>

        <Button onClick={generateCandidates}>Generate from Wikidata</Button>

        <List>
          {candidates.map(c => (
            <ListItem key={c.wikidata_id}>
              <Checkbox
                checked={c.isSelected}
                onChange={e => {
                  setCandidates(candidates.map(candidate =>
                    candidate.wikidata_id === c.wikidata_id
                      ? {...candidate, isSelected: e.target.checked}
                      : candidate
                  ));
                }}
              />
              <Typography>{c.full_name}</Typography>
              {c.exists_in_db ? (
                <Chip label={`EXISTS (${c.existing_photo_count} photos)`} color="success" />
              ) : (
                <Chip label="NEW" color="primary" />
              )}
            </ListItem>
          ))}
        </List>

        <Button onClick={startTraining} variant="contained">
          Start Training {candidates.filter(c => c.isSelected).length} People
        </Button>
      </Box>
    );
  }

  if (step === 2) {
    return (
      <Box>
        <Typography variant="h4">Training in Progress</Typography>
        <Typography>Batch ID: {batchId}</Typography>

        {batchStatus && (
          <>
            <LinearProgress
              variant="determinate"
              value={batchStatus.progress_percentage}
            />
            <Typography>
              {batchStatus.completed}/{batchStatus.total} completed ({batchStatus.progress_percentage}%)
            </Typography>

            <List>
              {batchStatus.people.map(p => (
                <ListItem key={p.folder_name}>
                  <Typography>
                    {p.status === 'completed' && '✅'}
                    {p.status === 'processing' && '🔄'}
                    {p.status === 'queued' && '⏳'}
                    {p.status === 'failed' && '❌'}
                    {' '}
                    {p.full_name} - {p.status.toUpperCase()}
                    {p.photos_downloaded > 0 && ` (${p.photos_downloaded} photos)`}
                  </Typography>
                </ListItem>
              ))}
            </List>
          </>
        )}
      </Box>
    );
  }

  // Step 3: Review & Deploy
  return (
    <Box>
      <Typography variant="h4">Review & Deploy</Typography>
      {/* Implementation similar to staging list */}
    </Box>
  );
}

export default TrainingWorkflow;
```

---

## Key Differences from Old System

| Feature | Old System (OpenAI) | New System (Wikidata) |
|---------|---------------------|----------------------|
| Celebrity Source | GPT-4.1 generates names | Wikipedia structured data |
| Cost | ~$0.02 per occupation | FREE |
| Accuracy | Sometimes generates fake people | Always real people |
| Coverage | 20-30 names per occupation | 100-500 names per occupation |
| DB Check | ❌ No check | ✅ Checks before showing user |
| Batch Processing | ❌ One at a time | ✅ Batch with progress tracking |
| Deployment | ❌ Manual script | ✅ API endpoint with UI |

---

## Testing the Workflow

### Step-by-step Test:

1. **Get countries:** `GET /api/training/countries`
2. **Get occupations:** `GET /api/training/occupations`
3. **Generate candidates:** `POST /api/training/generate-candidates`
   ```json
   {"country": "serbia", "occupation": "actor"}
   ```
4. **Start training:** `POST /api/training/start-batch`
   ```json
   {"candidates": [...], "domain": "serbia"}
   ```
5. **Monitor progress:** `GET /api/training/batch/{batch_id}/status` (poll every 3s)
6. **Get staging list:** `GET /api/training/staging-list`
7. **Deploy:** `POST /api/training/deploy`
   ```json
   {"people": ["novak_djokovic"], "domain": "serbia"}
   ```

---

## Storage Structure

```
storage/
├── recognized_faces_prod/          # PRODUCTION
│   └── serbia/
│       ├── novak_djokovic/         # ✅ In production
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       └── ana_ivanovic/
│           └── ...
│
├── trainingPassSerbia/             # STAGING (waiting for deployment)
│   ├── milena_dravic/              # ⏳ Ready to deploy (28 photos)
│   ├── rade_serbedzija/            # ⏳ Ready to deploy (31 photos)
│   └── unknown_person/             # ❌ Will be deleted (3 photos < 5 minimum)
│
└── training_batches/               # Batch metadata
    ├── a1b2c3d4.json              # Batch progress tracking
    └── e5f6g7h8.json
```

---

## Error Handling

### Common Errors:

1. **No celebrities found**
   ```json
   {
     "success": false,
     "message": "No celebrities found in Wikidata",
     "candidates": []
   }
   ```
   **Fix:** Try different country/occupation combination

2. **Batch not found**
   ```json
   {
     "success": false,
     "message": "Batch not found"
   }
   ```
   **Fix:** Check batch_id is correct

3. **Person already in training**
   - Wikidata query returns person with `exists_in_db: true`
   - User should deselect or system should auto-deselect

4. **Too few images**
   ```json
   {
     "skipped": [
       {
         "folder": "unknown_person",
         "reason": "Too few images (3), minimum is 5"
       }
     ]
   }
   ```
   **Fix:** Folder automatically deleted, no action needed

---

## Performance Notes

- **Wikidata query:** 2-5 seconds for 500 celebrities
- **Candidate generation:** 3-7 seconds (includes DB check)
- **Image download per person:** 30-60 seconds
- **DeepFace validation:** 2-5 minutes per person (runs in background)
- **Total time for 20 people:** ~30-45 minutes

---

## Next Steps

1. Build the 3-page UI flow
2. Implement polling for batch status
3. Add photo preview functionality
4. Add search/autocomplete for custom person addition
5. Add statistics dashboard

For any questions, refer to the main API documentation at `TRAINING_UI_API_DOCS.md`.
