# A/B Testing UI - Frontend Developer Documentation

## Overview

This document describes a **separate UI** for the A/B testing system that compares two face recognition models side-by-side. This is **independent** of the training UI and focuses solely on testing and comparing recognition performance.

**Purpose**: Help stakeholders decide whether to migrate from the current system (VGG-Face) to the improved system (ArcFace).

---

## Quick Links

📚 **Prerequisites**: Read these first:
- `TRAINING_UI_API_DOCS.md` - Authentication & API basics (pages 1-20)
- `AB_TESTING_PLAN.md` - Complete A/B testing strategy
- `TRAINING_UI_QUICK_REFERENCE.md` - Quick API reference

**Authentication**: Same as training UI (see TRAINING_UI_API_DOCS.md page 6-11)

---

## What You're Building

A **comparison dashboard** with 4 main sections:

```
┌─────────────────────────────────────────────────────────────┐
│                   A/B TESTING DASHBOARD                      │
└─────────────────────────────────────────────────────────────┘

1. LIVE COMPARISON TOOL
   Upload image → See both systems side-by-side → Compare results

2. METRICS DASHBOARD
   Daily/weekly statistics → Agreement rate → Accuracy → Performance

3. DECISION SUPPORT
   Should we migrate? → Scorecard → Recommendations → Trends

4. TEST HISTORY
   Browse past comparisons → Filter by result type → Export data
```

---

## Architecture

### Two Pipelines Running in Parallel

```
                    User Uploads Image
                           ↓
            ┌──────────────┴──────────────┐
            ↓                              ↓
    ┌──────────────┐              ┌──────────────┐
    │  PIPELINE A  │              │  PIPELINE B  │
    │  (Current)   │              │  (Improved)  │
    ├──────────────┤              ├──────────────┤
    │ VGG-Face     │              │ ArcFace      │
    │ Threshold:   │              │ Threshold:   │
    │ 0.35         │              │ 0.50         │
    │ Confidence:  │              │ Confidence:  │
    │ 99.5%        │              │ 98%          │
    └──────────────┘              └──────────────┘
            ↓                              ↓
    ┌──────────────┐              ┌──────────────┐
    │  Result A    │              │  Result B    │
    │  Person: X   │              │  Person: X   │
    │  Conf: 93.5% │              │  Conf: 97.2% │
    │  Time: 2.1s  │              │  Time: 2.4s  │
    └──────────────┘              └──────────────┘
            │                              │
            └──────────────┬───────────────┘
                           ↓
                  ┌────────────────┐
                  │  COMPARISON    │
                  │  - Agreement?  │
                  │  - Accuracy?   │
                  │  - Performance?│
                  └────────────────┘
```

---

## API Endpoints (Reference)

All details in `TRAINING_UI_API_DOCS.md`. Quick summary:

### 1. Run Comparison Test
```http
POST /api/test/recognize
Authorization: Bearer <token>
Content-Type: multipart/form-data

FormData:
  - image: <file>
  - image_id: <string> (optional)
  - ground_truth: <string> (optional)

Response: See "Data Models" section below
```

### 2. Get Daily Metrics
```http
GET /api/test/metrics/daily?date=2025-01-13
Authorization: Bearer <token>

Response: See TRAINING_UI_API_DOCS.md page 73-75
```

### 3. Get Weekly Metrics
```http
GET /api/test/metrics/weekly
Authorization: Bearer <token>

Response: See TRAINING_UI_API_DOCS.md page 76-78
```

### 4. Health Check
```http
GET /api/test/health
Authorization: Bearer <token>
```

---

## Data Models

### Comparison Result
```typescript
interface ComparisonResult {
  image_id: string;
  ground_truth?: string;

  // Pipeline A (Current System)
  pipeline_a_result: {
    status: 'success' | 'no_faces' | 'error';
    person?: string;
    confidence?: number;
    processing_time: number;
    profile_used: {
      name: string;
      model: string;
      threshold: number;
      detection_confidence: number;
    };
  };

  // Pipeline B (Improved System)
  pipeline_b_result: {
    status: 'success' | 'no_faces' | 'error';
    person?: string;
    confidence?: number;
    processing_time: number;
    profile_used: {
      name: string;
      model: string;
      threshold: number;
      detection_confidence: number;
    };
  };

  // Comparison Metrics
  comparison: {
    comparison_id: string;
    comparison_metrics: {
      both_succeeded: boolean;
      both_failed: boolean;
      only_a_succeeded: boolean;
      only_b_succeeded: boolean;
      results_match: boolean;
      confidence_difference?: number;  // B - A
      processing_time_difference?: number;  // B - A
      faster_pipeline?: 'pipeline_a' | 'pipeline_b';
      accuracy?: {
        pipeline_a_correct: boolean;
        pipeline_b_correct: boolean;
        winner: 'pipeline_a' | 'pipeline_b' | 'both' | 'neither';
      };
    };
  };

  recommendation: string;
}
```

### Metrics Summary
```typescript
interface MetricsSummary {
  total_comparisons: number;
  date_range: {
    start: string;
    end: string;
  };

  status_breakdown: {
    both_succeeded: { count: number; percentage: number };
    both_failed: { count: number; percentage: number };
    only_a_succeeded: { count: number; percentage: number };
    only_b_succeeded: { count: number; percentage: number };
  };

  agreement: {
    total_agreements: number;
    total_disagreements: number;
    agreement_rate: number;  // percentage
  };

  accuracy?: {
    total_with_ground_truth: number;
    pipeline_a_accuracy: number;  // percentage
    pipeline_b_accuracy: number;  // percentage
    improvement: number;  // percentage difference
  };

  performance: {
    avg_confidence_difference?: number;
    avg_time_difference_ms?: number;
    pipeline_b_faster_count: number;
  };
}
```

---

## UI Design & Components

### Page 1: Live Comparison Tool

**URL**: `/ab-testing/compare`

**Layout**:
```
┌────────────────────────────────────────────────────────────┐
│  A/B Testing - Live Comparison                              │
└────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  Upload Test Image                      │
│  ┌───────────────────────────────────┐  │
│  │                                   │  │
│  │   Drop image here or click        │  │
│  │   [Browse]                        │  │
│  │                                   │  │
│  └───────────────────────────────────┘  │
│                                         │
│  Optional:                              │
│  Ground Truth: [                    ]  │
│  Image ID: [                        ]  │
│                                         │
│  [Run Comparison Test]                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Results                                                     │
├───────────────────────────┬─────────────────────────────────┤
│  PIPELINE A               │  PIPELINE B                     │
│  Current System           │  Improved System                │
├───────────────────────────┼─────────────────────────────────┤
│  Model: VGG-Face          │  Model: ArcFace                 │
│  Threshold: 0.35          │  Threshold: 0.50                │
│  Confidence: 99.5%        │  Confidence: 98%                │
├───────────────────────────┼─────────────────────────────────┤
│  Status: ✓ Success        │  Status: ✓ Success              │
│  Person: John Doe         │  Person: John Doe               │
│  Confidence: 93.5%        │  Confidence: 97.2%              │
│  Time: 2.14s              │  Time: 2.45s                    │
└───────────────────────────┴─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Analysis                                                    │
├─────────────────────────────────────────────────────────────┤
│  ✓ Both systems agree on: John Doe                          │
│  ↑ Pipeline B has 3.7% higher confidence                    │
│  → Pipeline A is 0.31s faster                               │
│  ✓ Both systems are correct (vs ground truth)               │
│                                                             │
│  Recommendation:                                            │
│  Both agree. Pipeline B has 3.7% higher confidence.         │
└─────────────────────────────────────────────────────────────┘
```

**Component Structure**:
```jsx
<LiveComparisonPage>
  <ImageUploader
    onUpload={handleImageUpload}
    groundTruth={groundTruth}
    imageId={imageId}
  />

  {result && (
    <>
      <ResultsComparison
        pipelineA={result.pipeline_a_result}
        pipelineB={result.pipeline_b_result}
        comparison={result.comparison}
      />

      <AnalysisPanel
        comparison={result.comparison}
        recommendation={result.recommendation}
      />
    </>
  )}
</LiveComparisonPage>
```

---

### Page 2: Metrics Dashboard

**URL**: `/ab-testing/metrics`

**Layout**:
```
┌────────────────────────────────────────────────────────────┐
│  A/B Testing - Metrics Dashboard                           │
└────────────────────────────────────────────────────────────┘

Period: [● Daily  ○ Weekly]   Date: [2025-01-13 ▼]  [Refresh]

┌─────────────────────────────────────────────────────────────┐
│  Key Metrics                                                 │
├───────────┬───────────┬───────────┬───────────┬─────────────┤
│ Total     │ Agreement │ Pipeline B│ Pipeline B│ Improvement │
│ Tests     │ Rate      │ Accuracy  │ Faster    │             │
├───────────┼───────────┼───────────┼───────────┼─────────────┤
│    1,234  │   87.5%   │   95.2%   │    45%    │   +12.3%    │
│           │   ↑ 2.1%  │   ↑ 5.4%  │   ↓ 5%    │   ↑ 3.2%    │
└───────────┴───────────┴───────────┴───────────┴─────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Status Breakdown                                            │
│                                                             │
│  Both Succeeded        ████████████████████ 82.3% (1,015)  │
│  Only B Succeeded      ███ 8.5% (105)                      │
│  Only A Succeeded      ██ 5.2% (64)                        │
│  Both Failed           █ 4.0% (50)                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Accuracy Comparison (vs Ground Truth)                      │
│                                                             │
│  Pipeline A (Current)                                       │
│  Correct: 785    Wrong: 115    Accuracy: 87.2%             │
│  ████████████████████████████████████░░░░░                 │
│                                                             │
│  Pipeline B (Improved)                                      │
│  Correct: 892    Wrong: 8      Accuracy: 99.1%             │
│  ████████████████████████████████████████████████          │
│                                                             │
│  Improvement: +11.9%                                        │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Performance Metrics                                         │
│                                                             │
│  Average Confidence Difference: +5.2% (B is higher)         │
│  Average Time Difference: +0.28s (B is slower)              │
│  Pipeline B Faster Count: 556 / 1,234 (45%)                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Recommendations                                             │
│                                                             │
│  ✓ STRONG RECOMMENDATION: Pipeline B shows 11.9% accuracy  │
│    improvement. Consider migration.                         │
│                                                             │
│  ✓ Pipeline B finds faces that Pipeline A misses (8.5% vs  │
│    5.2%). Good sign!                                        │
│                                                             │
│  ✓ High agreement rate (87.5%). Pipelines are consistent.  │
│                                                             │
│  ⚠ Pipeline B is 0.28s slower. May need optimization.      │
└─────────────────────────────────────────────────────────────┘
```

**Component Structure**:
```jsx
<MetricsDashboard>
  <PeriodSelector
    period={period}
    date={date}
    onPeriodChange={handlePeriodChange}
  />

  <KeyMetricsCards metrics={summary} />

  <StatusBreakdownChart data={summary.status_breakdown} />

  <AccuracyComparisonChart
    pipelineA={summary.accuracy.pipeline_a_accuracy}
    pipelineB={summary.accuracy.pipeline_b_accuracy}
    improvement={summary.accuracy.improvement}
  />

  <PerformanceMetrics data={summary.performance} />

  <RecommendationsList recommendations={recommendations} />
</MetricsDashboard>
```

---

### Page 3: Decision Support

**URL**: `/ab-testing/decision`

**Layout**:
```
┌────────────────────────────────────────────────────────────┐
│  A/B Testing - Migration Decision Support                  │
└────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Decision Scorecard                                          │
├─────────────────────────────────────────────────────────────┤
│  Metric               Weight  │ Pipeline A │ Pipeline B │ ✓ │
├───────────────────────────────┼────────────┼────────────┼───┤
│  Accuracy             40%     │   87.2%    │   99.1%    │ B │
│  False Negative Rate  25%     │   12.8%    │   0.9%     │ B │
│  False Positive Rate  20%     │   2.3%     │   1.8%     │ B │
│  Processing Time      10%     │   2.14s    │   2.45s    │ A │
│  User Satisfaction    5%      │   8.2/10   │   9.1/10   │ B │
├───────────────────────────────┼────────────┼────────────┼───┤
│  Total Weighted Score 100%    │   72.4     │   88.9     │ B │
└─────────────────────────────────────────────────────────────┘

Score Difference: +16.5 points in favor of Pipeline B

┌─────────────────────────────────────────────────────────────┐
│  Decision Matrix                                             │
├─────────────────────────────────────────────────────────────┤
│  Criteria                          Status    Threshold       │
├─────────────────────────────────────────────────────────────┤
│  Accuracy improvement > 5%          ✓ Pass   (11.9% > 5%)  │
│  False negative rate decrease       ✓ Pass   (-11.9%)      │
│  No increase in false positives     ✓ Pass   (-0.5%)       │
│  No significant performance loss    ⚠ Review (+0.31s)      │
│  No critical bugs found             ✓ Pass   (0 bugs)      │
│  Minimum 1000 comparisons           ✓ Pass   (1,234)       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Recommendation                                              │
├─────────────────────────────────────────────────────────────┤
│  🎯 RECOMMENDED ACTION: MIGRATE TO PIPELINE B                │
│                                                             │
│  Confidence: HIGH                                            │
│  Score: 88.9 / 100 (Pipeline B wins by +16.5 points)       │
│                                                             │
│  ✓ Strong accuracy improvement (+11.9%)                     │
│  ✓ Significant reduction in false negatives (-11.9%)        │
│  ✓ No increase in false positives                          │
│  ⚠ Slightly slower (+0.31s) - acceptable trade-off         │
│  ✓ High user satisfaction                                   │
│                                                             │
│  Next Steps:                                                │
│  1. Begin gradual rollout (10% → 25% → 50% → 100%)        │
│  2. Monitor performance in production                       │
│  3. Keep rollback plan ready                               │
│                                                             │
│  [View Rollout Plan]  [Export Report]  [Mark Decision]     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Historical Trend (Last 4 Weeks)                            │
│                                                             │
│  Accuracy                          Agreement Rate           │
│  100% ┤                            100% ┤                   │
│       │        ╭──────B             95% │   ╭────╮          │
│   90% │   ╭───╯                     90% │  ╱      ╲         │
│       │ ─A────                      85% │╭╯        ╲        │
│   80% ├────────────────             80% ├────────────       │
│       Week 1   2   3   4                Week 1   2   3   4  │
└─────────────────────────────────────────────────────────────┘
```

**Component Structure**:
```jsx
<DecisionSupportPage>
  <DecisionScorecard
    metrics={scorecardData}
    weightedScores={weightedScores}
  />

  <DecisionMatrix
    criteria={decisionCriteria}
    thresholds={thresholds}
  />

  <RecommendationPanel
    recommendation={recommendation}
    confidence={confidence}
    nextSteps={nextSteps}
  />

  <HistoricalTrends
    accuracyTrend={accuracyData}
    agreementTrend={agreementData}
  />
</DecisionSupportPage>
```

---

### Page 4: Test History

**URL**: `/ab-testing/history`

**Layout**:
```
┌────────────────────────────────────────────────────────────┐
│  A/B Testing - Test History                                │
└────────────────────────────────────────────────────────────┘

Filters:
Result: [All ▼]  Status: [All ▼]  Date: [Last 7 days ▼]
Search: [                                    ] [Search]

┌─────────────────────────────────────────────────────────────┐
│  Image ID    │ Date/Time │ Ground Truth │ A Result │ B Result│ Agreement │ Winner │
├──────────────┼───────────┼──────────────┼──────────┼─────────┼───────────┼────────┤
│ test_001.jpg │ 01-13 14h │ John Doe     │ John Doe │ John Doe│ ✓ Agree   │ Both   │
│              │           │              │ 93.5%    │ 97.2%   │           │        │
├──────────────┼───────────┼──────────────┼──────────┼─────────┼───────────┼────────┤
│ test_002.jpg │ 01-13 14h │ Jane Smith   │ None     │ Jane S. │ ✗ Differ  │ B      │
│              │           │              │ No face  │ 89.3%   │           │        │
├──────────────┼───────────┼──────────────┼──────────┼─────────┼───────────┼────────┤
│ test_003.jpg │ 01-13 14h │ Bob Johnson  │ Bob J.   │ Bob J.  │ ✓ Agree   │ Both   │
│              │           │              │ 91.2%    │ 94.8%   │           │        │
├──────────────┼───────────┼──────────────┼──────────┼─────────┼───────────┼────────┤
│ ...          │ ...       │ ...          │ ...      │ ...     │ ...       │ ...    │
└─────────────────────────────────────────────────────────────┘

[< Previous]  Page 1 of 25  [Next >]

[Export to CSV]  [Export to JSON]
```

**Component Structure**:
```jsx
<TestHistoryPage>
  <FilterPanel
    resultFilter={resultFilter}
    statusFilter={statusFilter}
    dateFilter={dateFilter}
    searchQuery={searchQuery}
    onFilterChange={handleFilterChange}
  />

  <TestHistoryTable
    tests={tests}
    onTestClick={handleTestClick}
  />

  <Pagination
    page={page}
    totalPages={totalPages}
    onPageChange={handlePageChange}
  />

  <ExportButtons
    onExportCSV={handleExportCSV}
    onExportJSON={handleExportJSON}
  />
</TestHistoryPage>
```

---

## React Implementation Examples

### Hook: Use Comparison Test

```jsx
// hooks/useComparisonTest.js
import { useState } from 'react';
import api from '../api';

export function useComparisonTest() {
  const [testing, setTesting] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const runComparison = async (file, options = {}) => {
    setTesting(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      if (options.imageId) {
        formData.append('image_id', options.imageId);
      }

      if (options.groundTruth) {
        formData.append('ground_truth', options.groundTruth);
      }

      const { data } = await api.post('/api/test/recognize', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      setResult(data);
      return data;
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message;
      setError(errorMessage);
      throw err;
    } finally {
      setTesting(false);
    }
  };

  return { runComparison, testing, result, error };
}
```

### Hook: Use Metrics

```jsx
// hooks/useMetrics.js
import { useState, useEffect } from 'react';
import api from '../api';

export function useMetrics(period = 'daily', date = null) {
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  const fetchMetrics = async () => {
    setLoading(true);
    setError(null);

    try {
      const endpoint = period === 'daily'
        ? '/api/test/metrics/daily'
        : '/api/test/metrics/weekly';

      const params = date && period === 'daily' ? { date } : {};

      const { data } = await api.get(endpoint, { params });

      setMetrics(data.summary);
      return data;
    } catch (err) {
      const errorMessage = err.response?.data?.error || err.message;
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
  }, [period, date]);

  return { metrics, loading, error, refetch: fetchMetrics };
}
```

### Component: Results Comparison Panel

```jsx
// components/ResultsComparison.jsx
import React from 'react';
import { Box, Card, Grid, Typography, Chip } from '@mui/material';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

function ResultsComparison({ pipelineA, pipelineB, comparison }) {
  const getStatusIcon = (status) => {
    return status === 'success'
      ? <CheckCircleIcon color="success" />
      : <ErrorIcon color="error" />;
  };

  const getStatusColor = (status) => {
    return status === 'success' ? 'success' : 'error';
  };

  return (
    <Box sx={{ my: 3 }}>
      <Typography variant="h5" gutterBottom>
        Results Comparison
      </Typography>

      <Grid container spacing={2}>
        {/* Pipeline A */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, borderLeft: '4px solid #1976d2' }}>
            <Typography variant="h6" gutterBottom>
              Pipeline A - Current System
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Model: {pipelineA.profile_used.model}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Threshold: {pipelineA.profile_used.threshold}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence: {pipelineA.profile_used.detection_confidence * 100}%
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              {getStatusIcon(pipelineA.status)}
              <Typography sx={{ ml: 1 }}>
                Status: {pipelineA.status}
              </Typography>
            </Box>

            {pipelineA.status === 'success' && (
              <>
                <Typography variant="h6" sx={{ mt: 2 }}>
                  Person: {pipelineA.person}
                </Typography>
                <Typography variant="body1">
                  Confidence: {pipelineA.confidence}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Processing Time: {pipelineA.processing_time.toFixed(2)}s
                </Typography>
              </>
            )}
          </Card>
        </Grid>

        {/* Pipeline B */}
        <Grid item xs={12} md={6}>
          <Card sx={{ p: 2, borderLeft: '4px solid #2e7d32' }}>
            <Typography variant="h6" gutterBottom>
              Pipeline B - Improved System
            </Typography>

            <Box sx={{ mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                Model: {pipelineB.profile_used.model}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Threshold: {pipelineB.profile_used.threshold}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Confidence: {pipelineB.profile_used.detection_confidence * 100}%
              </Typography>
            </Box>

            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              {getStatusIcon(pipelineB.status)}
              <Typography sx={{ ml: 1 }}>
                Status: {pipelineB.status}
              </Typography>
            </Box>

            {pipelineB.status === 'success' && (
              <>
                <Typography variant="h6" sx={{ mt: 2 }}>
                  Person: {pipelineB.person}
                </Typography>
                <Typography variant="body1">
                  Confidence: {pipelineB.confidence}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Processing Time: {pipelineB.processing_time.toFixed(2)}s
                </Typography>
              </>
            )}
          </Card>
        </Grid>
      </Grid>

      {/* Comparison Metrics */}
      <Card sx={{ p: 2, mt: 2, bgcolor: '#f5f5f5' }}>
        <Typography variant="h6" gutterBottom>
          Analysis
        </Typography>

        {comparison.comparison_metrics.both_succeeded && (
          <>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
              {comparison.comparison_metrics.results_match ? (
                <>
                  <CheckCircleIcon color="success" sx={{ mr: 1 }} />
                  <Typography>
                    Both systems agree on: {pipelineA.person}
                  </Typography>
                </>
              ) : (
                <>
                  <ErrorIcon color="warning" sx={{ mr: 1 }} />
                  <Typography>
                    Systems disagree: A says "{pipelineA.person}", B says "{pipelineB.person}"
                  </Typography>
                </>
              )}
            </Box>

            {comparison.comparison_metrics.confidence_difference && (
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUpIcon sx={{ mr: 1 }} />
                <Typography>
                  Confidence difference: {comparison.comparison_metrics.confidence_difference > 0 ? '+' : ''}
                  {comparison.comparison_metrics.confidence_difference}%
                  {' '}(Pipeline B {comparison.comparison_metrics.confidence_difference > 0 ? 'higher' : 'lower'})
                </Typography>
              </Box>
            )}

            {comparison.comparison_metrics.faster_pipeline && (
              <Typography variant="body2" color="text.secondary">
                Faster: {comparison.comparison_metrics.faster_pipeline === 'pipeline_a' ? 'Pipeline A' : 'Pipeline B'}
                {' '}by {Math.abs(comparison.comparison_metrics.processing_time_difference * 1000).toFixed(0)}ms
              </Typography>
            )}

            {comparison.comparison_metrics.accuracy && (
              <Box sx={{ mt: 2 }}>
                <Typography variant="body2" gutterBottom>
                  Accuracy vs Ground Truth:
                </Typography>
                <Chip
                  label={`Pipeline A: ${comparison.comparison_metrics.accuracy.pipeline_a_correct ? '✓ Correct' : '✗ Wrong'}`}
                  color={comparison.comparison_metrics.accuracy.pipeline_a_correct ? 'success' : 'error'}
                  size="small"
                  sx={{ mr: 1 }}
                />
                <Chip
                  label={`Pipeline B: ${comparison.comparison_metrics.accuracy.pipeline_b_correct ? '✓ Correct' : '✗ Wrong'}`}
                  color={comparison.comparison_metrics.accuracy.pipeline_b_correct ? 'success' : 'error'}
                  size="small"
                />
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Winner: {comparison.comparison_metrics.accuracy.winner.toUpperCase()}
                </Typography>
              </Box>
            )}
          </>
        )}

        {comparison.comparison_metrics.only_b_succeeded && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <CheckCircleIcon color="success" sx={{ mr: 1 }} />
            <Typography>
              Only Pipeline B found a face: {pipelineB.person}
            </Typography>
          </Box>
        )}

        {comparison.comparison_metrics.only_a_succeeded && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ErrorIcon color="warning" sx={{ mr: 1 }} />
            <Typography>
              Only Pipeline A found a face: {pipelineA.person}
            </Typography>
          </Box>
        )}

        {comparison.comparison_metrics.both_failed && (
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <ErrorIcon color="error" sx={{ mr: 1 }} />
            <Typography>
              Both pipelines failed to recognize face
            </Typography>
          </Box>
        )}
      </Card>
    </Box>
  );
}

export default ResultsComparison;
```

### Component: Key Metrics Cards

```jsx
// components/KeyMetricsCards.jsx
import React from 'react';
import { Grid, Card, CardContent, Typography, Box } from '@mui/material';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';
import TrendingDownIcon from '@mui/icons-material/TrendingDown';

function MetricCard({ title, value, change, isPositive }) {
  return (
    <Card>
      <CardContent>
        <Typography color="text.secondary" gutterBottom>
          {title}
        </Typography>
        <Typography variant="h4" component="div">
          {value}
        </Typography>
        {change && (
          <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
            {isPositive ? (
              <TrendingUpIcon color="success" fontSize="small" />
            ) : (
              <TrendingDownIcon color="error" fontSize="small" />
            )}
            <Typography
              variant="body2"
              color={isPositive ? 'success.main' : 'error.main'}
              sx={{ ml: 0.5 }}
            >
              {change}
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
}

function KeyMetricsCards({ metrics }) {
  if (!metrics) return null;

  return (
    <Grid container spacing={3} sx={{ mb: 3 }}>
      <Grid item xs={12} sm={6} md={2.4}>
        <MetricCard
          title="Total Tests"
          value={metrics.total_comparisons.toLocaleString()}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={2.4}>
        <MetricCard
          title="Agreement Rate"
          value={`${metrics.agreement.agreement_rate}%`}
          change="↑ 2.1%"
          isPositive={true}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={2.4}>
        <MetricCard
          title="Pipeline B Accuracy"
          value={`${metrics.accuracy?.pipeline_b_accuracy || 0}%`}
          change={metrics.accuracy?.improvement ? `↑ ${metrics.accuracy.improvement}%` : null}
          isPositive={true}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={2.4}>
        <MetricCard
          title="Pipeline B Faster"
          value={`${((metrics.performance.pipeline_b_faster_count / metrics.total_comparisons) * 100).toFixed(1)}%`}
        />
      </Grid>

      <Grid item xs={12} sm={6} md={2.4}>
        <MetricCard
          title="Improvement"
          value={`+${metrics.accuracy?.improvement || 0}%`}
          change={metrics.accuracy?.improvement > 10 ? "Strong!" : "Good"}
          isPositive={true}
        />
      </Grid>
    </Grid>
  );
}

export default KeyMetricsCards;
```

---

## Visualization Libraries

### Recommended Charts

**Chart Library**: Recharts, Chart.js, or Victory

```bash
npm install recharts
# or
npm install react-chartjs-2 chart.js
```

### Example: Status Breakdown Chart

```jsx
// components/StatusBreakdownChart.jsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';

function StatusBreakdownChart({ data }) {
  const chartData = [
    { name: 'Both Succeeded', count: data.both_succeeded.count, percentage: data.both_succeeded.percentage },
    { name: 'Only B Succeeded', count: data.only_b_succeeded.count, percentage: data.only_b_succeeded.percentage },
    { name: 'Only A Succeeded', count: data.only_a_succeeded.count, percentage: data.only_a_succeeded.percentage },
    { name: 'Both Failed', count: data.both_failed.count, percentage: data.both_failed.percentage },
  ];

  return (
    <BarChart width={600} height={300} data={chartData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Legend />
      <Bar dataKey="count" fill="#1976d2" />
    </BarChart>
  );
}

export default StatusBreakdownChart;
```

### Example: Accuracy Comparison Chart

```jsx
// components/AccuracyComparisonChart.jsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell } from 'recharts';

function AccuracyComparisonChart({ pipelineA, pipelineB, improvement }) {
  const data = [
    { name: 'Pipeline A (Current)', accuracy: pipelineA, fill: '#1976d2' },
    { name: 'Pipeline B (Improved)', accuracy: pipelineB, fill: '#2e7d32' }
  ];

  return (
    <div>
      <BarChart width={600} height={300} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis domain={[0, 100]} />
        <Tooltip />
        <Legend />
        <Bar dataKey="accuracy" label={{ position: 'top' }}>
          {data.map((entry, index) => (
            <Cell key={`cell-${index}`} fill={entry.fill} />
          ))}
        </Bar>
      </BarChart>
      <Typography variant="h6" align="center" sx={{ mt: 2 }}>
        Improvement: +{improvement}%
      </Typography>
    </div>
  );
}

export default AccuracyComparisonChart;
```

---

## Export Functionality

### Export to CSV

```javascript
// utils/exportUtils.js
export function exportToCSV(data, filename) {
  const headers = [
    'Image ID',
    'Date',
    'Ground Truth',
    'Pipeline A Person',
    'Pipeline A Confidence',
    'Pipeline B Person',
    'Pipeline B Confidence',
    'Agreement',
    'Winner'
  ];

  const rows = data.map(item => [
    item.image_id,
    new Date(item.comparison.timestamp).toLocaleString(),
    item.ground_truth || 'N/A',
    item.pipeline_a_result.person || 'None',
    item.pipeline_a_result.confidence || 'N/A',
    item.pipeline_b_result.person || 'None',
    item.pipeline_b_result.confidence || 'N/A',
    item.comparison.comparison_metrics.results_match ? 'Yes' : 'No',
    item.comparison.comparison_metrics.accuracy?.winner || 'N/A'
  ]);

  const csvContent = [
    headers.join(','),
    ...rows.map(row => row.join(','))
  ].join('\n');

  const blob = new Blob([csvContent], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
}
```

### Export to JSON

```javascript
export function exportToJSON(data, filename) {
  const jsonContent = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonContent], { type: 'application/json' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
}
```

---

## Routing Structure

```javascript
// App.js or routes.js
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import LiveComparisonPage from './pages/LiveComparisonPage';
import MetricsDashboard from './pages/MetricsDashboard';
import DecisionSupportPage from './pages/DecisionSupportPage';
import TestHistoryPage from './pages/TestHistoryPage';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/ab-testing" element={<Layout />}>
          <Route path="compare" element={<LiveComparisonPage />} />
          <Route path="metrics" element={<MetricsDashboard />} />
          <Route path="decision" element={<DecisionSupportPage />} />
          <Route path="history" element={<TestHistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
```

---

## Navigation Menu

```jsx
// components/Layout.jsx
import { Outlet, Link } from 'react-router-dom';
import { AppBar, Toolbar, Typography, Button, Box } from '@mui/material';

function Layout() {
  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            A/B Testing Dashboard
          </Typography>
          <Button color="inherit" component={Link} to="/ab-testing/compare">
            Live Test
          </Button>
          <Button color="inherit" component={Link} to="/ab-testing/metrics">
            Metrics
          </Button>
          <Button color="inherit" component={Link} to="/ab-testing/decision">
            Decision
          </Button>
          <Button color="inherit" component={Link} to="/ab-testing/history">
            History
          </Button>
        </Toolbar>
      </AppBar>

      <Box sx={{ p: 3 }}>
        <Outlet />
      </Box>
    </>
  );
}

export default Layout;
```

---

## Polling for Real-Time Updates

```javascript
// hooks/usePolling.js
import { useEffect, useRef } from 'react';

export function usePolling(callback, interval = 5000, enabled = true) {
  const savedCallback = useRef();

  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled) return;

    function tick() {
      savedCallback.current();
    }

    const id = setInterval(tick, interval);
    return () => clearInterval(id);
  }, [interval, enabled]);
}
```

Usage:
```jsx
function MetricsDashboard() {
  const { metrics, refetch } = useMetrics();

  // Poll every 30 seconds
  usePolling(refetch, 30000, true);

  return <div>...</div>;
}
```

---

## Testing Checklist

### Manual Testing
- [ ] Upload image and run comparison
- [ ] View results side-by-side
- [ ] Check agreement/disagreement cases
- [ ] Verify confidence differences displayed
- [ ] Test with ground truth input
- [ ] View daily metrics
- [ ] View weekly metrics
- [ ] Check all chart visualizations
- [ ] Test decision scorecard calculation
- [ ] Export to CSV works
- [ ] Export to JSON works
- [ ] Pagination works in history
- [ ] Filters work in history
- [ ] Search works in history
- [ ] All error states handled
- [ ] Loading states shown

### Edge Cases
- [ ] Both systems fail
- [ ] Only A succeeds
- [ ] Only B succeeds
- [ ] Systems disagree on result
- [ ] No metrics available yet
- [ ] Empty history
- [ ] Invalid image upload

---

## Deployment Notes

### Environment Variables
```env
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_POLLING_INTERVAL=30000
```

### Build for Production
```bash
npm run build
```

### Serve Separately
This UI should be served separately from the training UI:
```
Training UI:  https://training.yourdomain.com
A/B Testing:  https://ab-testing.yourdomain.com
```

Or as separate routes on same domain:
```
/training/*     → Training UI
/ab-testing/*   → A/B Testing UI
```

---

## Summary

You're building a **decision support dashboard** with 4 main sections:

1. **Live Comparison** - Upload → See both systems → Compare
2. **Metrics Dashboard** - Statistics, charts, trends
3. **Decision Support** - Scorecard, recommendation, next steps
4. **Test History** - Browse, filter, export

**Key Features:**
- Side-by-side comparison of both models
- Visual metrics and charts
- Automated recommendations
- Export capabilities
- Real-time polling
- Decision scorecard

**Implementation Time:** ~2-3 weeks

**Complexity:** Medium (good component library helps a lot)

All API details are in `TRAINING_UI_API_DOCS.md` pages 70-85 and authentication is covered on pages 6-11.
