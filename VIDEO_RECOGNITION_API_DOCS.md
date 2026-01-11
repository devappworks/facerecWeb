# Video Face Recognition - Frontend Developer Guide

## Overview

This API allows you to upload videos and perform face recognition on extracted frames. The system extracts 1 frame every **3 seconds** (configurable) and identifies faces in each frame.

**Key Features:**
- ✅ Extract frames at configurable intervals (default: 3 seconds)
- ✅ Recognize faces in each extracted frame
- ✅ System performance monitoring (CPU, memory usage)
- ✅ Detailed logging of extraction and recognition
- ✅ Synchronous or asynchronous processing
- ✅ Support for multiple video formats (MP4, AVI, MOV, MKV, WebM, FLV, WMV)

---

## Table of Contents

1. [API Endpoints](#api-endpoints)
2. [Processing Workflow](#processing-workflow)
3. [Request/Response Examples](#requestresponse-examples)
4. [React Implementation](#react-implementation)
5. [Performance Monitoring](#performance-monitoring)
6. [Error Handling](#error-handling)
7. [Testing Guide](#testing-guide)

---

## API Endpoints

### 1. Upload Video (Synchronous)

**POST** `/api/video/upload`

Upload and process video immediately. Wait for complete results.

**Request:**
- Content-Type: `multipart/form-data`
- Fields:
  - `file` (required): Video file
  - `domain` (optional): Domain for recognition (default: "serbia")
  - `interval_seconds` (optional): Frame extraction interval (default: 3.0)

**Response** (200 OK):
```json
{
  "success": true,
  "video_id": "a1b2c3d4e5f6",
  "domain": "serbia",
  "processed_at": "2025-11-14T12:00:00",
  "extraction_info": {
    "total_frames": 900,
    "extracted_count": 30,
    "extraction_time": 5.2,
    "video_info": {
      "fps": 30.0,
      "duration": 30.5,
      "width": 1920,
      "height": 1080
    }
  },
  "statistics": {
    "total_frames": 30,
    "recognized_frames": 18,
    "failed_frames": 0,
    "recognition_rate": 60.0,
    "unique_persons": 3,
    "persons_list": ["Novak Djokovic", "Ana Ivanovic", "Emir Kusturica"]
  },
  "performance": {
    "processing_time_seconds": 45.3,
    "frames_per_second": 0.66,
    "avg_cpu_percent": 78.5,
    "memory_used_mb": 1250.4,
    "final_memory_mb": 2100.8
  },
  "results": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "filename": "frame_000000_t0.00s.jpg",
      "recognized": true,
      "person": "Novak Djokovic",
      "confidence": 99.2
    },
    {
      "frame_number": 90,
      "timestamp": 3.0,
      "filename": "frame_000090_t3.00s.jpg",
      "recognized": true,
      "person": "Ana Ivanovic",
      "confidence": 98.5
    }
  ]
}
```

---

### 2. Upload Video (Asynchronous)

**POST** `/api/video/upload-async`

Upload video and process in background. Returns immediately with video_id.

**Request:**
- Same as synchronous upload

**Response** (202 Accepted):
```json
{
  "success": true,
  "message": "Video uploaded successfully. Processing in background.",
  "video_id": "a1b2c3d4e5f6",
  "status_endpoint": "/api/video/status/a1b2c3d4e5f6"
}
```

---

### 3. Get Video Status

**GET** `/api/video/status/{video_id}`

Check processing status and get results.

**Response - Processing** (202 Accepted):
```json
{
  "success": false,
  "video_id": "a1b2c3d4e5f6",
  "status": "processing",
  "message": "Video is still being processed"
}
```

**Response - Completed** (200 OK):
```json
{
  "success": true,
  "video_id": "a1b2c3d4e5f6",
  "status": "completed",
  ... (same as synchronous upload response)
}
```

**Response - Not Found** (404):
```json
{
  "success": false,
  "video_id": "a1b2c3d4e5f6",
  "status": "not_found",
  "message": "Video not found"
}
```

---

### 4. Get API Info

**GET** `/api/video/info`

Get supported formats and limits.

**Response** (200 OK):
```json
{
  "success": true,
  "supported_formats": ["mp4", "avi", "mov", "mkv", "webm", "flv", "wmv"],
  "max_file_size_mb": 100,
  "default_interval_seconds": 3.0,
  "min_interval_seconds": 0.1,
  "max_interval_seconds": 60.0,
  "endpoints": {
    "upload": "/api/video/upload",
    "upload_async": "/api/video/upload-async",
    "status": "/api/video/status/{video_id}",
    "info": "/api/video/info"
  }
}
```

---

## Processing Workflow

### Synchronous Processing

```
User uploads video
      ↓
Server receives video → Saves to storage
      ↓
Extract frames (1 per 3 seconds)
      ↓
Run face recognition on each frame
      ↓
Generate statistics and performance metrics
      ↓
Return complete results (all at once)
```

**Use when:**
- Video is short (< 1 minute)
- User can wait for results
- Simpler frontend implementation

### Asynchronous Processing

```
User uploads video
      ↓
Server receives video → Saves to storage
      ↓
Return video_id immediately
      ↓
[Background thread]
      ↓
Extract frames (1 per 3 seconds)
      ↓
Run face recognition on each frame
      ↓
Save results to JSON file
      ↓
[Frontend polls status endpoint every 3s]
      ↓
Return results when complete
```

**Use when:**
- Video is long (> 1 minute)
- Better user experience (no long wait)
- Can show progress/loading indicator

---

## Request/Response Examples

### Example 1: Upload Short Video (Synchronous)

```javascript
const formData = new FormData();
formData.append('file', videoFile);
formData.append('domain', 'serbia');
formData.append('interval_seconds', '3');

const response = await fetch('/api/video/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();

if (result.success) {
  console.log(`Processed ${result.statistics.total_frames} frames`);
  console.log(`Recognized ${result.statistics.recognized_frames} frames`);
  console.log(`Found ${result.statistics.unique_persons} unique persons:`);
  console.log(result.statistics.persons_list);

  // Display results
  result.results.forEach(frame => {
    if (frame.recognized) {
      console.log(`${frame.timestamp}s: ${frame.person} (${frame.confidence}%)`);
    }
  });
}
```

### Example 2: Upload Long Video (Asynchronous with Polling)

```javascript
// 1. Upload video
const formData = new FormData();
formData.append('file', videoFile);
formData.append('domain', 'serbia');
formData.append('interval_seconds', '3');

const uploadResponse = await fetch('/api/video/upload-async', {
  method: 'POST',
  body: formData
});

const uploadResult = await uploadResponse.json();

if (uploadResult.success) {
  const videoId = uploadResult.video_id;

  // 2. Poll for status
  const checkStatus = async () => {
    const statusResponse = await fetch(`/api/video/status/${videoId}`);
    const statusResult = await statusResponse.json();

    if (statusResponse.status === 202) {
      // Still processing
      console.log('Still processing...');
      setTimeout(checkStatus, 3000); // Check again in 3 seconds
    } else if (statusResult.success) {
      // Completed!
      console.log('Processing complete!');
      console.log(`Found ${statusResult.statistics.unique_persons} persons`);
      // Display results...
    } else {
      // Error
      console.error('Processing failed:', statusResult.message);
    }
  };

  checkStatus();
}
```

---

## React Implementation

### Complete Video Upload Component

```jsx
import React, { useState, useEffect } from 'react';
import {
  Box, Button, LinearProgress, Typography, Card, CardContent,
  List, ListItem, Chip, Grid
} from '@mui/material';
import { CloudUpload } from '@mui/icons-material';

function VideoRecognitionPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [videoId, setVideoId] = useState(null);
  const [result, setResult] = useState(null);
  const [intervalSeconds, setIntervalSeconds] = useState(3);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
      setVideoId(null);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;

    setProcessing(true);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('domain', 'serbia');
      formData.append('interval_seconds', intervalSeconds);

      // Use async upload for better UX
      const response = await fetch('/api/video/upload-async', {
        method: 'POST',
        body: formData
      });

      const uploadResult = await response.json();

      if (uploadResult.success) {
        setVideoId(uploadResult.video_id);
        // Start polling
      } else {
        alert(`Upload failed: ${uploadResult.message}`);
        setProcessing(false);
      }
    } catch (error) {
      alert(`Error: ${error.message}`);
      setProcessing(false);
    }
  };

  // Poll for status when videoId is set
  useEffect(() => {
    if (!videoId) return;

    const pollStatus = async () => {
      try {
        const response = await fetch(`/api/video/status/${videoId}`);
        const statusResult = await response.json();

        if (response.status === 202) {
          // Still processing, poll again in 3 seconds
          setTimeout(pollStatus, 3000);
        } else if (statusResult.success) {
          // Completed!
          setResult(statusResult);
          setProcessing(false);
        } else {
          // Error
          alert(`Processing failed: ${statusResult.message}`);
          setProcessing(false);
        }
      } catch (error) {
        alert(`Error checking status: ${error.message}`);
        setProcessing(false);
      }
    };

    pollStatus();
  }, [videoId]);

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Video Face Recognition
      </Typography>

      {/* Upload Section */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <input
            type="file"
            accept="video/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="video-upload"
          />
          <label htmlFor="video-upload">
            <Button
              variant="contained"
              component="span"
              startIcon={<CloudUpload />}
              disabled={processing}
            >
              Select Video
            </Button>
          </label>

          {selectedFile && (
            <Box sx={{ mt: 2 }}>
              <Typography>
                Selected: {selectedFile.name} ({(selectedFile.size / (1024 * 1024)).toFixed(2)} MB)
              </Typography>

              <Box sx={{ mt: 2 }}>
                <Typography>Frame Extraction Interval:</Typography>
                <input
                  type="number"
                  value={intervalSeconds}
                  onChange={(e) => setIntervalSeconds(parseFloat(e.target.value))}
                  min="0.1"
                  max="60"
                  step="0.5"
                  style={{ padding: '8px', fontSize: '16px' }}
                />
                <Typography variant="caption" sx={{ ml: 1 }}>
                  seconds (1 frame every {intervalSeconds}s)
                </Typography>
              </Box>

              <Button
                variant="contained"
                color="primary"
                onClick={handleUpload}
                disabled={processing}
                sx={{ mt: 2 }}
              >
                {processing ? 'Processing...' : 'Upload and Process'}
              </Button>
            </Box>
          )}

          {processing && (
            <Box sx={{ mt: 3 }}>
              <LinearProgress />
              <Typography sx={{ mt: 1 }}>
                Processing video... This may take a few minutes.
              </Typography>
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Results Section */}
      {result && (
        <>
          {/* Statistics Card */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recognition Statistics
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Total Frames Extracted
                  </Typography>
                  <Typography variant="h4">
                    {result.statistics.total_frames}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Frames Recognized
                  </Typography>
                  <Typography variant="h4" color="success.main">
                    {result.statistics.recognized_frames}
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Recognition Rate
                  </Typography>
                  <Typography variant="h4">
                    {result.statistics.recognition_rate}%
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Unique Persons
                  </Typography>
                  <Typography variant="h4">
                    {result.statistics.unique_persons}
                  </Typography>
                </Grid>
              </Grid>

              <Box sx={{ mt: 3 }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Detected Persons:
                </Typography>
                {result.statistics.persons_list.map((person) => (
                  <Chip
                    key={person}
                    label={person}
                    color="primary"
                    sx={{ mr: 1, mb: 1 }}
                  />
                ))}
              </Box>
            </CardContent>
          </Card>

          {/* Performance Card */}
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Processing Time
                  </Typography>
                  <Typography variant="h6">
                    {result.performance.processing_time_seconds}s
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Frames Per Second
                  </Typography>
                  <Typography variant="h6">
                    {result.performance.frames_per_second.toFixed(2)} FPS
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Avg CPU Usage
                  </Typography>
                  <Typography variant="h6">
                    {result.performance.avg_cpu_percent.toFixed(1)}%
                  </Typography>
                </Grid>
                <Grid item xs={12} md={3}>
                  <Typography variant="body2" color="text.secondary">
                    Memory Used
                  </Typography>
                  <Typography variant="h6">
                    {result.performance.memory_used_mb.toFixed(1)} MB
                  </Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>

          {/* Frame Results List */}
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Frame-by-Frame Results ({result.results.length} frames)
              </Typography>
              <List>
                {result.results.map((frame, index) => (
                  <ListItem key={index} divider>
                    <Grid container spacing={2} alignItems="center">
                      <Grid item xs={2}>
                        <Typography variant="body2">
                          {frame.timestamp.toFixed(2)}s
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        {frame.recognized ? (
                          <>
                            <Typography variant="body1">
                              {frame.person}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              Confidence: {frame.confidence}%
                            </Typography>
                          </>
                        ) : (
                          <Typography variant="body1" color="text.secondary">
                            Unknown
                          </Typography>
                        )}
                      </Grid>
                      <Grid item xs={4}>
                        <Chip
                          label={frame.recognized ? 'Recognized' : 'Unknown'}
                          color={frame.recognized ? 'success' : 'default'}
                          size="small"
                        />
                      </Grid>
                    </Grid>
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </>
      )}
    </Box>
  );
}

export default VideoRecognitionPage;
```

---

## Performance Monitoring

The API automatically tracks system performance during processing:

### Metrics Tracked

1. **Processing Time** - Total time to process video
2. **Frames Per Second** - Processing speed (frames/second)
3. **CPU Usage** - Average CPU utilization during processing
4. **Memory Usage** - Memory consumed during processing

### Interpreting Performance Data

```javascript
// Good Performance
{
  "processing_time_seconds": 30.5,
  "frames_per_second": 1.5,      // Processing > 1 FPS = good
  "avg_cpu_percent": 60.2,       // < 80% = healthy
  "memory_used_mb": 500.4        // Reasonable memory usage
}

// Slow Performance (might need optimization)
{
  "processing_time_seconds": 120.8,
  "frames_per_second": 0.25,     // < 0.5 FPS = slow
  "avg_cpu_percent": 95.8,       // > 90% = overloaded
  "memory_used_mb": 3500.2       // High memory usage
}
```

### Logging Messages

Backend logs detailed information:

```
INFO: Received video upload: my_video.mp4, size: 25.50 MB, domain: serbia, interval: 3.0s
INFO: Video: a1b2c3d4e5f6, FPS: 30.0, Total frames: 900, Duration: 30.00s, Extracting every 90 frames (3.0s)
INFO: Extracted 10 frames...
INFO: Extracted 20 frames...
INFO: Extracted 30 frames from 900 total in 5.20s
INFO: Starting recognition on 30 frames for video a1b2c3d4e5f6
INFO: Processed 10/30 frames | Recognized: 6 | CPU: 75.2% | Memory: 1850.4 MB
INFO: Processed 20/30 frames | Recognized: 12 | CPU: 78.5% | Memory: 1950.8 MB
INFO: Processed 30/30 frames | Recognized: 18 | CPU: 76.1% | Memory: 2050.2 MB
INFO: Video a1b2c3d4e5f6 processing complete:
  Frames processed: 30
  Recognized: 18 (60.0%)
  Unique persons: 3
  Processing time: 45.30s (0.66 FPS)
  CPU usage: 78.5%
  Memory used: 1250.4 MB
```

---

## Error Handling

### Common Errors

#### 1. File Too Large
```json
{
  "success": false,
  "message": "Video too large. Maximum size: 100 MB"
}
```
**HTTP Status:** 413 Payload Too Large

**Solution:** Compress video or increase MAX_VIDEO_SIZE_MB in config

#### 2. Invalid File Type
```json
{
  "success": false,
  "message": "Invalid file type. Allowed: mp4, avi, mov, mkv, webm, flv, wmv"
}
```
**HTTP Status:** 400 Bad Request

**Solution:** Convert video to supported format

#### 3. Invalid Interval
```json
{
  "success": false,
  "message": "interval_seconds must be between 0.1 and 60"
}
```
**HTTP Status:** 400 Bad Request

**Solution:** Use interval between 0.1 and 60 seconds

#### 4. Video Not Found
```json
{
  "success": false,
  "video_id": "abc123",
  "status": "not_found",
  "message": "Video not found"
}
```
**HTTP Status:** 404 Not Found

**Solution:** Check video_id is correct

### Frontend Error Handling Example

```javascript
const handleUpload = async () => {
  try {
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('domain', 'serbia');
    formData.append('interval_seconds', intervalSeconds);

    const response = await fetch('/api/video/upload-async', {
      method: 'POST',
      body: formData
    });

    const result = await response.json();

    if (!result.success) {
      // Handle API error
      switch (response.status) {
        case 400:
          alert(`Invalid request: ${result.message}`);
          break;
        case 413:
          alert('Video file is too large. Please upload a smaller video.');
          break;
        case 500:
          alert('Server error. Please try again later.');
          break;
        default:
          alert(`Error: ${result.message}`);
      }
      return;
    }

    // Success - start polling
    setVideoId(result.video_id);

  } catch (error) {
    // Handle network error
    alert(`Network error: ${error.message}`);
  }
};
```

---

## Testing Guide

### Manual Testing with cURL

#### 1. Check API Info
```bash
curl http://localhost:5000/api/video/info
```

#### 2. Upload Video (Synchronous)
```bash
curl -X POST http://localhost:5000/api/video/upload \
  -F "file=@/path/to/video.mp4" \
  -F "domain=serbia" \
  -F "interval_seconds=3"
```

#### 3. Upload Video (Asynchronous)
```bash
# Upload
curl -X POST http://localhost:5000/api/video/upload-async \
  -F "file=@/path/to/video.mp4" \
  -F "domain=serbia" \
  -F "interval_seconds=3"

# Get video_id from response, then check status
curl http://localhost:5000/api/video/status/a1b2c3d4e5f6
```

### Testing with Postman

1. **Create new request:** POST to `http://localhost:5000/api/video/upload-async`
2. **Body:** Form-data
   - Key: `file`, Type: File, Value: Select video file
   - Key: `domain`, Type: Text, Value: `serbia`
   - Key: `interval_seconds`, Type: Text, Value: `3`
3. **Send request**
4. **Copy video_id** from response
5. **Create new request:** GET to `http://localhost:5000/api/video/status/{video_id}`
6. **Poll every 3 seconds** until status is "completed"

### Expected Processing Times

| Video Duration | Frames Extracted (3s interval) | Processing Time | Notes |
|----------------|-------------------------------|-----------------|-------|
| 30 seconds | 10 frames | ~15-30 seconds | Fast |
| 1 minute | 20 frames | ~30-60 seconds | Moderate |
| 5 minutes | 100 frames | ~2-5 minutes | Slow, use async |
| 10 minutes | 200 frames | ~5-10 minutes | Very slow, use async |

**Rule of thumb:** Processing time ≈ 1-2 seconds per frame extracted

---

## Best Practices

### 1. Use Async for Long Videos
```javascript
// Videos > 1 minute: Use async
if (videoDuration > 60) {
  useAsyncUpload();
} else {
  useSyncUpload();
}
```

### 2. Show Progress Indicator
```jsx
{processing && (
  <Box>
    <LinearProgress />
    <Typography>
      Processing video... Extracted {extractedFrames} frames.
      Recognized {recognizedFrames} faces so far.
    </Typography>
  </Box>
)}
```

### 3. Validate File Before Upload
```javascript
const validateVideo = (file) => {
  // Check size
  const maxSizeMB = 100;
  if (file.size > maxSizeMB * 1024 * 1024) {
    alert(`File too large. Maximum: ${maxSizeMB} MB`);
    return false;
  }

  // Check format
  const allowedFormats = ['mp4', 'avi', 'mov', 'mkv', 'webm'];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowedFormats.includes(ext)) {
    alert(`Invalid format. Allowed: ${allowedFormats.join(', ')}`);
    return false;
  }

  return true;
};
```

### 4. Handle Network Errors
```javascript
const uploadWithRetry = async (formData, maxRetries = 3) => {
  for (let i = 0; i < maxRetries; i++) {
    try {
      const response = await fetch('/api/video/upload-async', {
        method: 'POST',
        body: formData
      });
      return await response.json();
    } catch (error) {
      if (i === maxRetries - 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
};
```

---

## Summary

**Key Points:**
- ✅ Extract 1 frame every 3 seconds (configurable)
- ✅ Use `/api/video/upload` for short videos (< 1 min)
- ✅ Use `/api/video/upload-async` for long videos (> 1 min)
- ✅ Poll `/api/video/status/{video_id}` every 3 seconds for async
- ✅ System automatically monitors CPU, memory, and performance
- ✅ Detailed logging shows extraction and recognition progress
- ✅ Support for MP4, AVI, MOV, MKV, WebM, FLV, WMV formats

**Ready to Implement:**
- Complete React component provided
- All error handling covered
- Performance monitoring included
- Testing guide with examples

For any questions, refer to this documentation or check the backend logs for detailed debugging information.
