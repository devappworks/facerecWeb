# A/B Testing Plan: Face Recognition Improvements

## Executive Summary

This document outlines a comprehensive plan to test the recommended face recognition improvements alongside the current system, enabling data-driven decision making before full deployment.

**Testing Approach**: Dual-Pipeline A/B Testing
- **Pipeline A**: Current system (VGG-Face, threshold 0.35, confidence 99.5%)
- **Pipeline B**: State-of-the-art system (ArcFace, threshold 0.50, confidence 99.5%)

**IMPORTANT UPDATE (2025-11-20)**: Replaced Facenet512 with ArcFace
- **Reason**: Facenet512 was 10-40x slower than VGG-Face, causing regular timeouts
- **ArcFace Benefits**: 99.8% LFW accuracy, 17ms inference (comparable to VGG-Face), production-ready

**Testing Duration**: 2-4 weeks
**Decision Criteria**: Compare accuracy, false positives/negatives, processing time
**Rollout Strategy**: Gradual migration based on test results

---

## Table of Contents

1. [Testing Architecture](#1-testing-architecture)
2. [Implementation Plan](#2-implementation-plan)
3. [Test Configuration Management](#3-test-configuration-management)
4. [Comparison & Logging System](#4-comparison--logging-system)
5. [Metrics & Reporting](#5-metrics--reporting)
6. [Test Dataset Preparation](#6-test-dataset-preparation)
7. [Testing Phases](#7-testing-phases)
8. [Decision Framework](#8-decision-framework)
9. [Migration Strategy](#9-migration-strategy)
10. [Code Implementation](#10-code-implementation)

---

## 1. Testing Architecture

### Option A: Dual-Endpoint Approach (Recommended)

**Architecture:**

```
                    ┌─────────────────────────────────────┐
                    │      Client Request                 │
                    │   "Test this image"                 │
                    └──────────┬──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌───────────────────────┐    ┌───────────────────────┐
    │   /recognize          │    │   /recognize-test     │
    │   (Current System)    │    │   (New System)        │
    ├───────────────────────┤    ├───────────────────────┤
    │ VGG-Face              │    │ Facenet512            │
    │ Threshold: 0.35       │    │ Threshold: 0.40       │
    │ Confidence: 99.5%     │    │ Confidence: 98%       │
    └──────────┬────────────┘    └──────────┬────────────┘
               │                             │
               │                             │
               ▼                             ▼
    ┌───────────────────────┐    ┌───────────────────────┐
    │   Result A            │    │   Result B            │
    └──────────┬────────────┘    └──────────┬────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                 ┌────────────────────────────┐
                 │   Comparison Service       │
                 │   - Log both results       │
                 │   - Calculate differences  │
                 │   - Store metrics          │
                 └──────────┬─────────────────┘
                            │
                            ▼
                 ┌────────────────────────────┐
                 │   Metrics Database         │
                 │   - Accuracy tracking      │
                 │   - Performance metrics    │
                 │   - User feedback          │
                 └────────────────────────────┘
```

**Pros:**
- ✅ No impact on production endpoint
- ✅ Easy to compare results side-by-side
- ✅ Can test independently
- ✅ Safe rollback

**Cons:**
- ⚠️ Requires additional endpoint
- ⚠️ Duplicate processing (2x compute)

---

### Option B: Smart Router Approach (Alternative)

**Architecture:**

```
                    ┌─────────────────────────────────────┐
                    │      Client Request                 │
                    └──────────┬──────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────────────────────┐
                    │   Smart Router Middleware           │
                    │   - Check test flag                 │
                    │   - Route 10% to new system         │
                    │   - Route 90% to current system     │
                    └──────────┬──────────────────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
                ▼                             ▼
    ┌───────────────────────┐    ┌───────────────────────┐
    │   Pipeline A          │    │   Pipeline B          │
    │   (90% traffic)       │    │   (10% traffic)       │
    └──────────┬────────────┘    └──────────┬────────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
                 ┌────────────────────────────┐
                 │   Return Result + Metrics  │
                 └────────────────────────────┘
```

**Pros:**
- ✅ Gradual rollout
- ✅ Real production traffic
- ✅ Single endpoint

**Cons:**
- ⚠️ More complex
- ⚠️ Risk to production

**Recommendation**: Use **Option A (Dual-Endpoint)** for initial testing, then **Option B (Smart Router)** for gradual rollout.

---

## 2. Implementation Plan

### Phase 1: Setup (Days 1-3)

**Week 1, Days 1-3: Infrastructure Setup**

#### Day 1: Create Test Configuration System
- [x] Create configuration profiles for A/B testing
- [x] Add new test endpoint
- [x] Set up logging infrastructure
- [x] Create metrics database tables

#### Day 2: Implement Comparison Service
- [x] Build result comparison logic
- [x] Create metrics tracking service
- [x] Set up logging middleware
- [x] Create test harness

#### Day 3: Testing Framework
- [x] Create test dataset
- [x] Build automated test runner
- [x] Set up metrics dashboard
- [x] Test infrastructure end-to-end

---

### Phase 2: Controlled Testing (Days 4-10)

**Week 1, Days 4-7: Internal Testing**
- Run automated tests with known dataset
- Compare results manually
- Fix any issues
- Validate metrics collection

**Week 2, Days 8-10: Beta Testing**
- Share test endpoint with small user group
- Collect feedback
- Monitor metrics
- Adjust configurations if needed

---

### Phase 3: Parallel Production Testing (Days 11-28)

**Weeks 2-4: Production A/B Testing**
- Deploy test endpoint to production
- Route 10-20% of traffic to test endpoint
- Collect comparative metrics
- Weekly analysis and reporting

---

### Phase 4: Decision & Migration (Days 29-35)

**Week 5: Analysis & Decision**
- Comprehensive data analysis
- Go/No-Go decision
- Migration planning
- Documentation updates

---

## 3. Test Configuration Management

### Configuration Profiles

**File**: `app/config/recognition_profiles.py`

```python
"""
Face recognition configuration profiles for A/B testing
"""

class RecognitionProfile:
    """Base class for recognition configuration"""

    def __init__(self, name, description):
        self.name = name
        self.description = description

    def get_config(self):
        raise NotImplementedError


class CurrentSystemProfile(RecognitionProfile):
    """
    Current production configuration (Pipeline A)
    """

    def __init__(self):
        super().__init__(
            name="current_system",
            description="Current VGG-Face based system"
        )

    def get_config(self):
        return {
            # Model configuration
            "model_name": "VGG-Face",
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds
            "recognition_threshold": 0.35,
            "detection_confidence_threshold": 0.995,  # 99.5%

            # Quality validation
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Metadata
            "profile_version": "1.0",
            "created": "2025-01-13",
            "is_production": True
        }


class ImprovedSystemProfile(RecognitionProfile):
    """
    Improved configuration based on research (Pipeline B)
    """

    def __init__(self):
        super().__init__(
            name="improved_system",
            description="Improved Facenet512 based system"
        )

    def get_config(self):
        return {
            # Model configuration
            "model_name": "Facenet512",  # CHANGED
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds
            "recognition_threshold": 0.40,  # CHANGED (was 0.35)
            "detection_confidence_threshold": 0.98,  # CHANGED (was 0.995)

            # Quality validation (same)
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Metadata
            "profile_version": "1.0",
            "created": "2025-01-13",
            "is_production": False,
            "is_test": True
        }


class EnsembleSystemProfile(RecognitionProfile):
    """
    Multi-model ensemble configuration (Pipeline C - Future)
    """

    def __init__(self):
        super().__init__(
            name="ensemble_system",
            description="Multi-model ensemble for maximum accuracy"
        )

    def get_config(self):
        return {
            # Model configuration
            "models": ["Facenet512", "ArcFace"],  # Multiple models
            "detector_backend": "retinaface",
            "distance_metric": "cosine",

            # Recognition thresholds (per model)
            "model_thresholds": {
                "Facenet512": 0.40,
                "ArcFace": 0.50
            },
            "detection_confidence_threshold": 0.98,

            # Quality validation
            "blur_threshold": 100,
            "contrast_threshold": 25,
            "brightness_min": 30,
            "brightness_max": 220,
            "edge_density_threshold": 15,

            # Processing options
            "enforce_detection": False,
            "normalize_face": True,
            "align": True,
            "batched": True,

            # Ensemble options
            "voting_strategy": "weighted",  # weighted, majority, confidence
            "min_model_agreement": 2,  # Minimum models that must agree

            # Metadata
            "profile_version": "1.0",
            "created": "2025-01-13",
            "is_production": False,
            "is_test": True
        }


class ProfileManager:
    """
    Manages recognition profiles
    """

    _profiles = {
        "current": CurrentSystemProfile(),
        "improved": ImprovedSystemProfile(),
        "ensemble": EnsembleSystemProfile()
    }

    @classmethod
    def get_profile(cls, name: str) -> RecognitionProfile:
        """Get profile by name"""
        if name not in cls._profiles:
            raise ValueError(f"Unknown profile: {name}. Available: {list(cls._profiles.keys())}")

        return cls._profiles[name]

    @classmethod
    def get_config(cls, name: str) -> dict:
        """Get configuration for profile"""
        profile = cls.get_profile(name)
        return profile.get_config()

    @classmethod
    def list_profiles(cls) -> list:
        """List all available profiles"""
        return [
            {
                "name": profile.name,
                "description": profile.description,
                "is_production": profile.get_config().get("is_production", False)
            }
            for profile in cls._profiles.values()
        ]
```

---

## 4. Comparison & Logging System

### Comparison Service

**File**: `app/services/comparison_service.py`

```python
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)


class ComparisonService:
    """
    Service to compare results from different recognition pipelines
    """

    @staticmethod
    def compare_results(result_a: Dict, result_b: Dict, image_id: str,
                       ground_truth: Optional[str] = None) -> Dict:
        """
        Compare two recognition results

        Args:
            result_a: Result from pipeline A (current system)
            result_b: Result from pipeline B (improved system)
            image_id: Unique identifier for the image
            ground_truth: Known correct answer (optional)

        Returns:
            Comparison report
        """
        comparison_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        comparison = {
            "comparison_id": comparison_id,
            "image_id": image_id,
            "timestamp": timestamp,
            "ground_truth": ground_truth,

            # Results from both pipelines
            "pipeline_a": {
                "name": "current_system",
                "status": result_a.get("status"),
                "person": result_a.get("person"),
                "confidence": ComparisonService._extract_confidence(result_a),
                "processing_time": ComparisonService._extract_processing_time(result_a),
                "recognized_persons_count": len(result_a.get("recognized_persons", []))
            },

            "pipeline_b": {
                "name": "improved_system",
                "status": result_b.get("status"),
                "person": result_b.get("person"),
                "confidence": ComparisonService._extract_confidence(result_b),
                "processing_time": ComparisonService._extract_processing_time(result_b),
                "recognized_persons_count": len(result_b.get("recognized_persons", []))
            },

            # Comparison metrics
            "comparison_metrics": {}
        }

        # Calculate comparison metrics
        comparison["comparison_metrics"] = ComparisonService._calculate_comparison_metrics(
            comparison["pipeline_a"],
            comparison["pipeline_b"],
            ground_truth
        )

        # Log comparison
        ComparisonService._log_comparison(comparison)

        # Store in database
        ComparisonService._store_comparison(comparison)

        return comparison

    @staticmethod
    def _extract_confidence(result: Dict) -> Optional[float]:
        """Extract confidence score from result"""
        try:
            if result.get("status") == "success":
                return result.get("best_match", {}).get(
                    "confidence_metrics", {}
                ).get("confidence_percentage")
        except:
            pass
        return None

    @staticmethod
    def _extract_processing_time(result: Dict) -> Optional[float]:
        """Extract processing time from result"""
        # Check various possible locations for processing time
        if "api_info" in result:
            return result["api_info"].get("request_processing_time")
        if "processing_time" in result:
            return result["processing_time"]
        return None

    @staticmethod
    def _calculate_comparison_metrics(pipeline_a: Dict, pipeline_b: Dict,
                                     ground_truth: Optional[str]) -> Dict:
        """
        Calculate comparison metrics between two pipelines
        """
        metrics = {
            "both_succeeded": False,
            "both_failed": False,
            "only_a_succeeded": False,
            "only_b_succeeded": False,
            "results_match": False,
            "confidence_difference": None,
            "processing_time_difference": None,
            "faster_pipeline": None
        }

        # Status comparison
        a_success = pipeline_a["status"] == "success"
        b_success = pipeline_b["status"] == "success"

        metrics["both_succeeded"] = a_success and b_success
        metrics["both_failed"] = not a_success and not b_success
        metrics["only_a_succeeded"] = a_success and not b_success
        metrics["only_b_succeeded"] = b_success and not a_success

        # Results comparison (if both succeeded)
        if metrics["both_succeeded"]:
            person_a = pipeline_a["person"]
            person_b = pipeline_b["person"]

            metrics["results_match"] = person_a == person_b

            # Confidence difference
            conf_a = pipeline_a["confidence"]
            conf_b = pipeline_b["confidence"]
            if conf_a is not None and conf_b is not None:
                metrics["confidence_difference"] = round(conf_b - conf_a, 2)

            # Processing time comparison
            time_a = pipeline_a["processing_time"]
            time_b = pipeline_b["processing_time"]
            if time_a is not None and time_b is not None:
                metrics["processing_time_difference"] = round(time_b - time_a, 3)
                metrics["faster_pipeline"] = "pipeline_a" if time_a < time_b else "pipeline_b"

        # Accuracy check (if ground truth provided)
        if ground_truth:
            metrics["accuracy"] = {
                "pipeline_a_correct": pipeline_a["person"] == ground_truth if a_success else False,
                "pipeline_b_correct": pipeline_b["person"] == ground_truth if b_success else False
            }

            # Determine winner
            if metrics["accuracy"]["pipeline_a_correct"] and not metrics["accuracy"]["pipeline_b_correct"]:
                metrics["accuracy"]["winner"] = "pipeline_a"
            elif metrics["accuracy"]["pipeline_b_correct"] and not metrics["accuracy"]["pipeline_a_correct"]:
                metrics["accuracy"]["winner"] = "pipeline_b"
            elif metrics["accuracy"]["pipeline_a_correct"] and metrics["accuracy"]["pipeline_b_correct"]:
                metrics["accuracy"]["winner"] = "both"
            else:
                metrics["accuracy"]["winner"] = "neither"

        return metrics

    @staticmethod
    def _log_comparison(comparison: Dict):
        """Log comparison results"""
        metrics = comparison["comparison_metrics"]

        log_msg = f"Comparison {comparison['comparison_id']}: "

        if metrics["both_succeeded"]:
            if metrics["results_match"]:
                log_msg += f"✅ Both agree: {comparison['pipeline_a']['person']}"
            else:
                log_msg += f"⚠️ Disagree: A={comparison['pipeline_a']['person']}, B={comparison['pipeline_b']['person']}"

            if metrics["confidence_difference"]:
                log_msg += f" | Conf diff: {metrics['confidence_difference']:+.1f}%"

        elif metrics["only_b_succeeded"]:
            log_msg += f"✨ Only B succeeded: {comparison['pipeline_b']['person']}"

        elif metrics["only_a_succeeded"]:
            log_msg += f"⚠️ Only A succeeded: {comparison['pipeline_a']['person']}"

        else:
            log_msg += "❌ Both failed"

        logger.info(log_msg)

    @staticmethod
    def _store_comparison(comparison: Dict):
        """
        Store comparison in database for later analysis

        TODO: Implement actual database storage
        For now, store in JSON file
        """
        try:
            import os
            os.makedirs("storage/comparisons", exist_ok=True)

            filename = f"storage/comparisons/{comparison['comparison_id']}.json"
            with open(filename, 'w') as f:
                json.dump(comparison, f, indent=2)

            logger.debug(f"Stored comparison: {filename}")

        except Exception as e:
            logger.error(f"Error storing comparison: {str(e)}")

    @staticmethod
    def get_comparison_summary(comparison_ids: Optional[List[str]] = None,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> Dict:
        """
        Get summary statistics from comparisons

        Args:
            comparison_ids: Specific comparison IDs to analyze
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)

        Returns:
            Summary statistics
        """
        import os
        import glob

        # Load comparison files
        comparisons = []
        comparison_files = glob.glob("storage/comparisons/*.json")

        for file_path in comparison_files:
            try:
                with open(file_path, 'r') as f:
                    comparison = json.load(f)

                # Filter by date if specified
                if start_date and comparison["timestamp"] < start_date:
                    continue
                if end_date and comparison["timestamp"] > end_date:
                    continue

                # Filter by IDs if specified
                if comparison_ids and comparison["comparison_id"] not in comparison_ids:
                    continue

                comparisons.append(comparison)

            except Exception as e:
                logger.error(f"Error loading comparison {file_path}: {str(e)}")

        if not comparisons:
            return {"error": "No comparisons found", "total": 0}

        # Calculate summary statistics
        total = len(comparisons)
        both_succeeded = sum(1 for c in comparisons if c["comparison_metrics"]["both_succeeded"])
        both_failed = sum(1 for c in comparisons if c["comparison_metrics"]["both_failed"])
        only_a_succeeded = sum(1 for c in comparisons if c["comparison_metrics"]["only_a_succeeded"])
        only_b_succeeded = sum(1 for c in comparisons if c["comparison_metrics"]["only_b_succeeded"])

        results_match = sum(1 for c in comparisons
                          if c["comparison_metrics"].get("results_match", False))

        # Accuracy statistics (if ground truth available)
        with_ground_truth = [c for c in comparisons if c.get("ground_truth")]
        accuracy_stats = None

        if with_ground_truth:
            a_correct = sum(1 for c in with_ground_truth
                          if c["comparison_metrics"].get("accuracy", {}).get("pipeline_a_correct", False))
            b_correct = sum(1 for c in with_ground_truth
                          if c["comparison_metrics"].get("accuracy", {}).get("pipeline_b_correct", False))

            accuracy_stats = {
                "total_with_ground_truth": len(with_ground_truth),
                "pipeline_a_accuracy": round(a_correct / len(with_ground_truth) * 100, 2),
                "pipeline_b_accuracy": round(b_correct / len(with_ground_truth) * 100, 2),
                "improvement": round((b_correct - a_correct) / len(with_ground_truth) * 100, 2)
            }

        # Performance statistics
        conf_diffs = [c["comparison_metrics"].get("confidence_difference")
                     for c in comparisons
                     if c["comparison_metrics"].get("confidence_difference") is not None]

        time_diffs = [c["comparison_metrics"].get("processing_time_difference")
                     for c in comparisons
                     if c["comparison_metrics"].get("processing_time_difference") is not None]

        summary = {
            "total_comparisons": total,
            "date_range": {
                "start": min(c["timestamp"] for c in comparisons),
                "end": max(c["timestamp"] for c in comparisons)
            },
            "status_breakdown": {
                "both_succeeded": {
                    "count": both_succeeded,
                    "percentage": round(both_succeeded / total * 100, 2)
                },
                "both_failed": {
                    "count": both_failed,
                    "percentage": round(both_failed / total * 100, 2)
                },
                "only_a_succeeded": {
                    "count": only_a_succeeded,
                    "percentage": round(only_a_succeeded / total * 100, 2)
                },
                "only_b_succeeded": {
                    "count": only_b_succeeded,
                    "percentage": round(only_b_succeeded / total * 100, 2)
                }
            },
            "agreement": {
                "total_agreements": results_match,
                "total_disagreements": both_succeeded - results_match,
                "agreement_rate": round(results_match / both_succeeded * 100, 2) if both_succeeded > 0 else 0
            },
            "accuracy": accuracy_stats,
            "performance": {
                "avg_confidence_difference": round(sum(conf_diffs) / len(conf_diffs), 2) if conf_diffs else None,
                "avg_time_difference_ms": round(sum(time_diffs) / len(time_diffs) * 1000, 2) if time_diffs else None,
                "pipeline_b_faster_count": sum(1 for c in comparisons
                                              if c["comparison_metrics"].get("faster_pipeline") == "pipeline_b")
            }
        }

        return summary
```

---

## 5. Metrics & Reporting

### Metrics to Track

#### Primary Metrics (Critical for Decision)

| Metric | Target | Threshold for Success |
|--------|--------|----------------------|
| **Accuracy** | >95% | >90% |
| **False Positive Rate** | <2% | <5% |
| **False Negative Rate** | <5% | <10% |
| **Agreement Rate** | N/A | Track trend |

#### Secondary Metrics (Performance)

| Metric | Target | Acceptable |
|--------|--------|------------|
| **Processing Time** | <3s | <5s |
| **Success Rate** | >90% | >80% |
| **Confidence Score** | >85% | >75% |

#### Comparison Metrics (A vs B)

| Metric | Good Indicator |
|--------|----------------|
| **B more accurate than A** | +10% improvement |
| **B finds faces A misses** | +15% recall |
| **B and A agree** | >80% agreement |
| **B faster than A** | Nice to have |

---

### Reporting Dashboard

**File**: `app/services/metrics_reporting_service.py`

```python
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from app.services.comparison_service import ComparisonService

logger = logging.getLogger(__name__)


class MetricsReportingService:
    """
    Generate reports and dashboards for A/B testing
    """

    @staticmethod
    def generate_daily_report(date: Optional[str] = None) -> Dict:
        """
        Generate daily comparison report

        Args:
            date: Date in ISO format (YYYY-MM-DD), defaults to today

        Returns:
            Daily report
        """
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        start_date = f"{date}T00:00:00"
        end_date = f"{date}T23:59:59"

        summary = ComparisonService.get_comparison_summary(
            start_date=start_date,
            end_date=end_date
        )

        report = {
            "report_type": "daily",
            "date": date,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "recommendations": MetricsReportingService._generate_recommendations(summary)
        }

        return report

    @staticmethod
    def generate_weekly_report() -> Dict:
        """Generate weekly comparison report"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=7)

        summary = ComparisonService.get_comparison_summary(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat()
        )

        report = {
            "report_type": "weekly",
            "period": {
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d")
            },
            "generated_at": datetime.utcnow().isoformat(),
            "summary": summary,
            "trends": MetricsReportingService._calculate_trends(summary),
            "recommendations": MetricsReportingService._generate_recommendations(summary)
        }

        return report

    @staticmethod
    def _generate_recommendations(summary: Dict) -> List[str]:
        """
        Generate recommendations based on metrics
        """
        recommendations = []

        if summary.get("error"):
            return ["Insufficient data for recommendations"]

        # Check accuracy improvement
        accuracy = summary.get("accuracy")
        if accuracy:
            improvement = accuracy.get("improvement", 0)

            if improvement > 10:
                recommendations.append(
                    f"✅ STRONG RECOMMENDATION: Pipeline B shows {improvement}% accuracy improvement. Consider migration."
                )
            elif improvement > 5:
                recommendations.append(
                    f"✅ Pipeline B shows {improvement}% accuracy improvement. Continue testing."
                )
            elif improvement > 0:
                recommendations.append(
                    f"⚠️ Pipeline B shows modest {improvement}% improvement. Collect more data."
                )
            elif improvement < -5:
                recommendations.append(
                    f"❌ WARNING: Pipeline B accuracy is {abs(improvement)}% worse. Review configuration."
                )

        # Check success rate
        status = summary.get("status_breakdown", {})
        only_b_succeeded = status.get("only_b_succeeded", {}).get("percentage", 0)
        only_a_succeeded = status.get("only_a_succeeded", {}).get("percentage", 0)

        if only_b_succeeded > only_a_succeeded + 10:
            recommendations.append(
                f"✅ Pipeline B finds faces that Pipeline A misses ({only_b_succeeded}% vs {only_a_succeeded}%). Good sign!"
            )
        elif only_a_succeeded > only_b_succeeded + 10:
            recommendations.append(
                f"⚠️ Pipeline A finds more faces than Pipeline B ({only_a_succeeded}% vs {only_b_succeeded}%). Investigate."
            )

        # Check agreement rate
        agreement = summary.get("agreement", {})
        agreement_rate = agreement.get("agreement_rate", 0)

        if agreement_rate > 90:
            recommendations.append(
                f"✅ High agreement rate ({agreement_rate}%). Pipelines are consistent."
            )
        elif agreement_rate < 70:
            recommendations.append(
                f"⚠️ Low agreement rate ({agreement_rate}%). Review disagreement cases."
            )

        # Check performance
        performance = summary.get("performance", {})
        time_diff = performance.get("avg_time_difference_ms")

        if time_diff and time_diff < -100:
            recommendations.append(
                f"✅ Pipeline B is {abs(time_diff):.0f}ms faster on average."
            )
        elif time_diff and time_diff > 500:
            recommendations.append(
                f"⚠️ Pipeline B is {time_diff:.0f}ms slower. May need optimization."
            )

        if not recommendations:
            recommendations.append("ℹ️ Continue collecting data for conclusive recommendations.")

        return recommendations

    @staticmethod
    def _calculate_trends(summary: Dict) -> Dict:
        """
        Calculate trends (requires historical data)
        TODO: Implement trend calculation with time series data
        """
        return {
            "accuracy_trend": "insufficient_data",
            "performance_trend": "insufficient_data",
            "note": "Trends require multiple days of data"
        }

    @staticmethod
    def generate_comparison_report(image_id: str, ground_truth: Optional[str] = None) -> str:
        """
        Generate human-readable comparison report for single image

        Args:
            image_id: Image identifier
            ground_truth: Known correct answer

        Returns:
            Formatted report string
        """
        # Load comparison data
        import glob
        comparison = None

        for file_path in glob.glob(f"storage/comparisons/*.json"):
            try:
                import json
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data.get("image_id") == image_id:
                        comparison = data
                        break
            except:
                continue

        if not comparison:
            return f"No comparison found for image: {image_id}"

        # Format report
        report_lines = [
            "="*60,
            f"COMPARISON REPORT: {image_id}",
            "="*60,
            ""
        ]

        if ground_truth:
            report_lines.append(f"Ground Truth: {ground_truth}")
            report_lines.append("")

        # Pipeline A results
        pipeline_a = comparison["pipeline_a"]
        report_lines.extend([
            "PIPELINE A (Current System - VGG-Face):",
            f"  Status: {pipeline_a['status']}",
            f"  Person: {pipeline_a.get('person', 'N/A')}",
            f"  Confidence: {pipeline_a.get('confidence', 'N/A')}%",
            f"  Processing Time: {pipeline_a.get('processing_time', 'N/A')}s",
            ""
        ])

        # Pipeline B results
        pipeline_b = comparison["pipeline_b"]
        report_lines.extend([
            "PIPELINE B (Improved System - Facenet512):",
            f"  Status: {pipeline_b['status']}",
            f"  Person: {pipeline_b.get('person', 'N/A')}",
            f"  Confidence: {pipeline_b.get('confidence', 'N/A')}%",
            f"  Processing Time: {pipeline_b.get('processing_time', 'N/A')}s",
            ""
        ])

        # Comparison metrics
        metrics = comparison["comparison_metrics"]
        report_lines.extend([
            "COMPARISON:",
        ])

        if metrics["both_succeeded"]:
            if metrics["results_match"]:
                report_lines.append("  ✅ Both pipelines agree")
            else:
                report_lines.append("  ⚠️ Pipelines disagree!")

            if metrics.get("confidence_difference"):
                diff = metrics["confidence_difference"]
                symbol = "+" if diff > 0 else ""
                report_lines.append(f"  Confidence Difference: {symbol}{diff}%")

            if metrics.get("faster_pipeline"):
                faster = "Pipeline B" if metrics["faster_pipeline"] == "pipeline_b" else "Pipeline A"
                time_diff = abs(metrics.get("processing_time_difference", 0))
                report_lines.append(f"  Faster: {faster} by {time_diff*1000:.0f}ms")

        elif metrics["only_b_succeeded"]:
            report_lines.append("  ✨ Only Pipeline B found a face")

        elif metrics["only_a_succeeded"]:
            report_lines.append("  ⚠️ Only Pipeline A found a face")

        else:
            report_lines.append("  ❌ Both pipelines failed")

        # Accuracy (if ground truth)
        if metrics.get("accuracy"):
            acc = metrics["accuracy"]
            report_lines.extend([
                "",
                "ACCURACY:",
                f"  Pipeline A: {'✅ Correct' if acc['pipeline_a_correct'] else '❌ Wrong'}",
                f"  Pipeline B: {'✅ Correct' if acc['pipeline_b_correct'] else '❌ Wrong'}",
                f"  Winner: {acc['winner'].upper()}"
            ])

        report_lines.extend([
            "",
            "="*60
        ])

        return "\n".join(report_lines)
```

---

## 6. Test Dataset Preparation

### Creating Ground Truth Dataset

**File**: `scripts/prepare_test_dataset.py`

```python
"""
Script to prepare test dataset with ground truth labels
"""

import os
import json
import shutil
from datetime import datetime


def create_test_dataset():
    """
    Create organized test dataset with ground truth

    Directory structure:
    storage/test_dataset/
    ├── ground_truth.json
    ├── images/
    │   ├── person1_001.jpg
    │   ├── person1_002.jpg
    │   ├── person2_001.jpg
    │   └── ...
    └── results/
        └── (comparison results will go here)
    """

    test_dataset_path = "storage/test_dataset"
    images_path = os.path.join(test_dataset_path, "images")
    results_path = os.path.join(test_dataset_path, "results")

    # Create directories
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)

    print("Test dataset structure created at:", test_dataset_path)
    print("")
    print("Next steps:")
    print("1. Add test images to:", images_path)
    print("2. Create ground_truth.json with format:")
    print("""
    {
      "images": [
        {
          "filename": "person1_001.jpg",
          "ground_truth": "John Doe",
          "description": "Frontal view, good lighting",
          "difficulty": "easy"
        },
        {
          "filename": "person2_001.jpg",
          "ground_truth": "Jane Smith",
          "description": "Side profile, dim lighting",
          "difficulty": "hard"
        }
      ]
    }
    """)
    print("3. Run test suite")


def load_ground_truth():
    """Load ground truth from JSON file"""
    ground_truth_file = "storage/test_dataset/ground_truth.json"

    if not os.path.exists(ground_truth_file):
        return {}

    with open(ground_truth_file, 'r') as f:
        data = json.load(f)

    # Create lookup dictionary
    lookup = {}
    for image_data in data.get("images", []):
        filename = image_data["filename"]
        lookup[filename] = {
            "ground_truth": image_data["ground_truth"],
            "description": image_data.get("description", ""),
            "difficulty": image_data.get("difficulty", "medium")
        }

    return lookup


if __name__ == "__main__":
    create_test_dataset()
```

### Test Dataset Categories

Organize test images by difficulty:

**Easy (Expected 95%+ accuracy)**:
- Frontal faces
- Good lighting
- High resolution
- No occlusion

**Medium (Expected 85-95% accuracy)**:
- Slightly angled faces
- Normal lighting
- Medium resolution
- Minimal occlusion

**Hard (Expected 70-85% accuracy)**:
- Profile views
- Poor lighting
- Low resolution
- Partial occlusion
- Motion blur

**Goal**: Both systems should perform well on easy/medium, improvement should show on hard cases.

---

## 7. Testing Phases

### Phase 1: Automated Testing (Week 1)

**Objective**: Validate both pipelines work correctly

**Activities**:
1. Run both pipelines on 100-200 test images with ground truth
2. Measure baseline metrics
3. Identify any bugs or configuration issues
4. Validate logging and comparison systems

**Success Criteria**:
- Both pipelines run without errors
- Comparison data is logged correctly
- Metrics can be generated

**Code**: See Section 10 for test runner implementation

---

### Phase 2: Beta Testing (Week 2)

**Objective**: Test with real users in controlled environment

**Activities**:
1. Deploy test endpoint to staging/beta environment
2. Share with 5-10 beta testers
3. Collect qualitative feedback
4. Monitor metrics

**Success Criteria**:
- No critical bugs reported
- Users can successfully use test endpoint
- Initial metrics look promising

**User Feedback Form**:
```
Beta Testing Feedback - Face Recognition A/B Test

Image tested: _______
Did the system recognize correctly? Yes / No / Unsure

If you know the correct answer:
- Current system (A) result: _______
- New system (B) result: _______
- Which was correct? A / B / Both / Neither

Quality rating:
- Image quality: Poor / Fair / Good / Excellent
- Lighting: Poor / Fair / Good / Excellent

Comments: _______

Would you trust the new system (B)? Yes / No / Needs improvement
```

---

### Phase 3: Production A/B Testing (Weeks 2-4)

**Objective**: Collect production data at scale

**Activities**:
1. Route 10-20% of production traffic to test endpoint
2. Collect minimum 1000 comparisons
3. Weekly analysis and reporting
4. Adjust configurations if needed

**Success Criteria**:
- Minimum 1000 comparisons collected
- No production incidents
- Clear trend in metrics
- Stakeholder confidence in data

**Weekly Review Checklist**:
- [ ] Generate weekly report
- [ ] Review accuracy metrics
- [ ] Check for anomalies
- [ ] Review disagreement cases
- [ ] Update stakeholders
- [ ] Document learnings

---

### Phase 4: Decision & Migration (Week 5)

**Objective**: Make data-driven go/no-go decision

**Activities**:
1. Comprehensive data analysis
2. Present findings to stakeholders
3. Make migration decision
4. Plan migration if approved

**Decision Criteria**:

**Go (Migrate to Pipeline B) if**:
- ✅ Accuracy improvement > 5%
- ✅ False negative rate decreases
- ✅ No increase in false positives
- ✅ No significant performance degradation
- ✅ No critical bugs found

**No-Go (Stay with Pipeline A) if**:
- ❌ Accuracy worse or no improvement
- ❌ Increased false positives
- ❌ Critical bugs found
- ❌ Significant performance issues

**Iterate (Adjust and Re-test) if**:
- ⚠️ Mixed results
- ⚠️ Needs parameter tuning
- ⚠️ Requires further optimization

---

## 8. Decision Framework

### Scorecard for Decision

| Metric | Weight | Pipeline A Score | Pipeline B Score | Winner |
|--------|--------|-----------------|-----------------|--------|
| **Accuracy** | 40% | ___ % | ___ % | ___ |
| **False Negative Rate** | 25% | ___ % | ___ % | ___ |
| **False Positive Rate** | 20% | ___ % | ___ % | ___ |
| **Processing Time** | 10% | ___ s | ___ s | ___ |
| **User Satisfaction** | 5% | ___ /10 | ___ /10 | ___ |
| **Total Weighted Score** | 100% | ___ | ___ | ___ |

**Decision Rule**:
- If B's weighted score > A's score by 10+ points: **MIGRATE**
- If B's weighted score > A's score by 5-10 points: **GRADUAL ROLLOUT**
- If B's weighted score > A's score by 0-5 points: **ITERATE & RE-TEST**
- If A's weighted score > B's score: **STAY WITH A**

---

## 9. Migration Strategy

### Gradual Rollout Plan

**If decision is GO for Pipeline B:**

**Week 1**: 10% traffic
- Route 10% of production to Pipeline B
- Monitor closely
- Be ready to rollback

**Week 2**: 25% traffic
- If no issues, increase to 25%
- Continue monitoring
- Collect feedback

**Week 3**: 50% traffic
- Increase to 50%
- Compare metrics with 100% Pipeline A baseline
- Validate improvements hold at scale

**Week 4**: 100% traffic
- Full migration to Pipeline B
- Keep Pipeline A code for rollback
- Update documentation

**Week 5**: Cleanup
- Remove Pipeline A code (keep in git history)
- Update all documentation
- Celebrate! 🎉

---

### Rollback Plan

**Trigger Conditions for Rollback**:
- Accuracy drops > 5%
- Error rate increases > 10%
- Critical bug discovered
- Performance degradation > 2x

**Rollback Procedure**:
1. Switch traffic back to Pipeline A immediately
2. Investigate root cause
3. Fix issue in Pipeline B
4. Re-test before attempting migration again

---

## 10. Code Implementation

### 10.1 Test Recognition Service

**File**: `app/services/test_recognition_service.py`

```python
"""
Test recognition service that runs both pipelines
"""

import logging
import time
from typing import Dict, Optional
from io import BytesIO

from app.services.recognition_service import RecognitionService
from app.config.recognition_profiles import ProfileManager
from app.services.comparison_service import ComparisonService

logger = logging.getLogger(__name__)


class TestRecognitionService:
    """
    Service to run recognition with different configurations for testing
    """

    @staticmethod
    def recognize_face_with_profile(image_bytes, domain: str, profile_name: str) -> Dict:
        """
        Run recognition with specific profile

        Args:
            image_bytes: Image data
            domain: Domain for database lookup
            profile_name: Profile to use (current, improved, ensemble)

        Returns:
            Recognition result
        """
        try:
            # Get configuration
            config = ProfileManager.get_config(profile_name)

            logger.info(f"Running recognition with profile: {profile_name}")
            logger.info(f"Model: {config.get('model_name')}, Threshold: {config.get('recognition_threshold')}")

            # Run recognition with this configuration
            result = RecognitionService.recognize_face_with_config(
                image_bytes, domain, config
            )

            # Add profile metadata to result
            result["profile_used"] = {
                "name": profile_name,
                "model": config.get("model_name"),
                "threshold": config.get("recognition_threshold"),
                "detection_confidence": config.get("detection_confidence_threshold")
            }

            return result

        except Exception as e:
            logger.error(f"Error in recognize_face_with_profile: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "profile_used": profile_name
            }

    @staticmethod
    def recognize_face_comparison(image_bytes, domain: str, image_id: Optional[str] = None,
                                  ground_truth: Optional[str] = None) -> Dict:
        """
        Run recognition with both current and improved profiles and compare

        Args:
            image_bytes: Image data
            domain: Domain for database lookup
            image_id: Unique identifier for this image
            ground_truth: Known correct answer (optional)

        Returns:
            Comparison results
        """
        if image_id is None:
            import uuid
            image_id = str(uuid.uuid4())

        logger.info(f"Starting comparison recognition for image: {image_id}")

        # Run Pipeline A (current system)
        logger.info("Running Pipeline A (current system)...")
        start_time_a = time.time()
        result_a = TestRecognitionService.recognize_face_with_profile(
            image_bytes, domain, "current"
        )
        time_a = time.time() - start_time_a
        result_a["processing_time"] = time_a

        # Run Pipeline B (improved system)
        logger.info("Running Pipeline B (improved system)...")
        start_time_b = time.time()

        # Need to reset BytesIO pointer if it was read
        if hasattr(image_bytes, 'seek'):
            image_bytes.seek(0)

        result_b = TestRecognitionService.recognize_face_with_profile(
            image_bytes, domain, "improved"
        )
        time_b = time.time() - start_time_b
        result_b["processing_time"] = time_b

        # Compare results
        comparison = ComparisonService.compare_results(
            result_a, result_b, image_id, ground_truth
        )

        # Combine into single response
        response = {
            "image_id": image_id,
            "ground_truth": ground_truth,
            "pipeline_a_result": result_a,
            "pipeline_b_result": result_b,
            "comparison": comparison,
            "recommendation": TestRecognitionService._get_recommendation(comparison)
        }

        return response

    @staticmethod
    def _get_recommendation(comparison: Dict) -> str:
        """
        Generate recommendation based on comparison
        """
        metrics = comparison["comparison_metrics"]

        if metrics["both_failed"]:
            return "Both pipelines failed to recognize face. Consider image quality."

        if metrics["only_b_succeeded"]:
            return "✨ Pipeline B found a face that Pipeline A missed!"

        if metrics["only_a_succeeded"]:
            return "⚠️ Pipeline A found a face but Pipeline B didn't. Review B configuration."

        if metrics["both_succeeded"]:
            if metrics["results_match"]:
                conf_diff = metrics.get("confidence_difference", 0)
                if conf_diff > 5:
                    return f"✅ Both agree. Pipeline B has {conf_diff}% higher confidence."
                elif conf_diff < -5:
                    return f"✅ Both agree. Pipeline A has {abs(conf_diff)}% higher confidence."
                else:
                    return "✅ Both pipelines agree with similar confidence."
            else:
                return "⚠️ Pipelines disagree on result. Manual review recommended."

        return "Unknown comparison state."
```

---

### 10.2 Update Recognition Service

Add new method to `app/services/recognition_service.py`:

```python
@staticmethod
def recognize_face_with_config(image_bytes, domain: str, config: Dict):
    """
    Run face recognition with custom configuration

    Args:
        image_bytes: Image data
        domain: Domain for database lookup
        config: Configuration dictionary with all parameters

    Returns:
        Recognition result
    """
    # Use config parameters instead of hardcoded values
    model_name = config.get("model_name", "VGG-Face")
    detector_backend = config.get("detector_backend", "retinaface")
    distance_metric = config.get("distance_metric", "cosine")
    recognition_threshold = config.get("recognition_threshold", 0.35)
    detection_confidence_threshold = config.get("detection_confidence_threshold", 0.995)

    # ... rest of recognition logic using these config values
    # (modify existing recognize_face method to use config)
```

---

### 10.3 Test Routes

**File**: `app/routes/test_recognition_routes.py`

```python
from flask import Blueprint, jsonify, request
from app.services.test_recognition_service import TestRecognitionService
from app.services.validation_service import ValidationService
from app.services.metrics_reporting_service import MetricsReportingService
import logging

test_recognition_routes = Blueprint('test_recognition', __name__)
logger = logging.getLogger(__name__)


@test_recognition_routes.route('/api/test/recognize', methods=['POST'])
def test_recognize():
    """
    Test endpoint that runs both pipelines and compares results
    """
    try:
        # Authentication
        auth_token = request.headers.get('Authorization')
        validation_service = ValidationService()

        if not auth_token:
            return jsonify({'message': 'Unauthorized'}), 401

        if not validation_service.validate_auth_token(auth_token):
            return jsonify({'message': 'Unauthorized'}), 401

        # Check for image
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image_file = request.files['image']
        if not image_file.filename:
            return jsonify({'error': 'No selected file'}), 400

        # Optional parameters
        image_id = request.form.get('image_id')
        ground_truth = request.form.get('ground_truth')  # For testing with known answers

        # Get domain
        domain = validation_service.get_domain()

        # Read image
        image_bytes = image_file.read()

        # Run comparison
        result = TestRecognitionService.recognize_face_comparison(
            image_bytes, domain, image_id, ground_truth
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in test_recognize endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@test_recognition_routes.route('/api/test/metrics/daily', methods=['GET'])
def get_daily_metrics():
    """
    Get daily metrics report
    """
    date = request.args.get('date')  # Optional: YYYY-MM-DD

    report = MetricsReportingService.generate_daily_report(date)

    return jsonify(report)


@test_recognition_routes.route('/api/test/metrics/weekly', methods=['GET'])
def get_weekly_metrics():
    """
    Get weekly metrics report
    """
    report = MetricsReportingService.generate_weekly_report()

    return jsonify(report)


@test_recognition_routes.route('/api/test/health', methods=['GET'])
def test_health():
    """
    Health check for test system
    """
    from app.config.recognition_profiles import ProfileManager

    return jsonify({
        "status": "operational",
        "available_profiles": ProfileManager.list_profiles(),
        "comparison_logging": "enabled",
        "metrics_reporting": "enabled"
    })
```

---

### 10.4 Test Runner Script

**File**: `scripts/run_ab_tests.py`

```python
"""
Automated test runner for A/B testing
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.test_recognition_service import TestRecognitionService
from app.services.comparison_service import ComparisonService
from scripts.prepare_test_dataset import load_ground_truth


def run_automated_tests():
    """
    Run automated tests on test dataset
    """
    print("="*60)
    print("AUTOMATED A/B TESTING")
    print("="*60)
    print()

    # Load test dataset
    test_images_path = "storage/test_dataset/images"
    ground_truth = load_ground_truth()

    if not os.path.exists(test_images_path):
        print(f"Error: Test images directory not found: {test_images_path}")
        print("Please run: python scripts/prepare_test_dataset.py")
        return

    # Get all test images
    image_files = [f for f in os.listdir(test_images_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print(f"No test images found in: {test_images_path}")
        return

    print(f"Found {len(image_files)} test images")
    print(f"Ground truth available for: {len(ground_truth)} images")
    print()

    # Run tests
    results = []
    for i, filename in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] Testing: {filename}")

        image_path = os.path.join(test_images_path, filename)

        # Read image
        with open(image_path, 'rb') as f:
            image_bytes = f.read()

        # Get ground truth if available
        gt = ground_truth.get(filename, {}).get("ground_truth")

        # Run comparison
        try:
            result = TestRecognitionService.recognize_face_comparison(
                image_bytes,
                domain="test",  # Use test domain
                image_id=filename,
                ground_truth=gt
            )

            results.append(result)

            # Print summary
            comp = result["comparison"]
            metrics = comp["comparison_metrics"]

            if metrics["both_succeeded"]:
                if metrics["results_match"]:
                    print(f"  ✅ Both agree: {result['pipeline_a_result']['person']}")
                else:
                    print(f"  ⚠️ Disagree: A={result['pipeline_a_result']['person']}, "
                          f"B={result['pipeline_b_result']['person']}")
            elif metrics["only_b_succeeded"]:
                print(f"  ✨ Only B succeeded: {result['pipeline_b_result']['person']}")
            elif metrics["only_a_succeeded"]:
                print(f"  ⚠️ Only A succeeded: {result['pipeline_a_result']['person']}")
            else:
                print(f"  ❌ Both failed")

            if gt:
                acc = metrics.get("accuracy", {})
                if acc:
                    winner = acc.get("winner", "unknown")
                    print(f"  Ground truth: {gt} | Winner: {winner}")

        except Exception as e:
            print(f"  ❌ Error: {str(e)}")

        print()
        time.sleep(0.5)  # Brief pause between tests

    # Generate summary
    print()
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)

    summary = ComparisonService.get_comparison_summary()

    print(f"Total tests: {summary['total_comparisons']}")
    print()

    status = summary["status_breakdown"]
    print("Status Breakdown:")
    print(f"  Both succeeded: {status['both_succeeded']['count']} ({status['both_succeeded']['percentage']}%)")
    print(f"  Both failed: {status['both_failed']['count']} ({status['both_failed']['percentage']}%)")
    print(f"  Only A succeeded: {status['only_a_succeeded']['count']} ({status['only_a_succeeded']['percentage']}%)")
    print(f"  Only B succeeded: {status['only_b_succeeded']['count']} ({status['only_b_succeeded']['percentage']}%)")
    print()

    agreement = summary["agreement"]
    print(f"Agreement rate: {agreement['agreement_rate']}%")
    print()

    if summary.get("accuracy"):
        acc = summary["accuracy"]
        print("Accuracy (vs ground truth):")
        print(f"  Pipeline A: {acc['pipeline_a_accuracy']}%")
        print(f"  Pipeline B: {acc['pipeline_b_accuracy']}%")
        print(f"  Improvement: {acc['improvement']:+.1f}%")
        print()

    # Save results
    results_file = "storage/test_dataset/results/automated_test_results.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            "test_run": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": summary,
            "individual_results": results
        }, f, indent=2)

    print(f"Results saved to: {results_file}")
    print("="*60)


if __name__ == "__main__":
    run_automated_tests()
```

---

## Summary & Action Items

### Immediate Next Steps (This Week)

#### Day 1: Setup Infrastructure
1. [ ] Create `app/config/recognition_profiles.py`
2. [ ] Create `app/services/comparison_service.py`
3. [ ] Create storage directories:
   ```bash
   mkdir -p storage/test_dataset/images
   mkdir -p storage/test_dataset/results
   mkdir -p storage/comparisons
   ```

#### Day 2: Implement Test Services
4. [ ] Create `app/services/test_recognition_service.py`
5. [ ] Create `app/services/metrics_reporting_service.py`
6. [ ] Update `app/services/recognition_service.py` with `recognize_face_with_config` method

#### Day 3: Add Test Endpoints
7. [ ] Create `app/routes/test_recognition_routes.py`
8. [ ] Register routes in `app/__init__.py`
9. [ ] Test endpoints with curl/Postman

#### Day 4-5: Prepare Test Dataset
10. [ ] Run `python scripts/prepare_test_dataset.py`
11. [ ] Add 50-100 test images with ground truth
12. [ ] Create `ground_truth.json` file

#### Day 6-7: Run Initial Tests
13. [ ] Run `python scripts/run_ab_tests.py`
14. [ ] Review results
15. [ ] Fix any issues
16. [ ] Generate first metrics report

---

### Testing Timeline

| Week | Phase | Activities | Deliverable |
|------|-------|-----------|-------------|
| **1** | Setup & Internal | Infrastructure, automated tests | Working test system |
| **2** | Beta | Beta user testing, collect feedback | Initial metrics |
| **3-4** | Production A/B | Route 10-20% traffic, collect data | Comprehensive data |
| **5** | Decision | Analysis, stakeholder review | Go/No-Go decision |

---

### Success Metrics

**Minimum Requirements for Migration**:
- ✅ 1000+ comparison data points collected
- ✅ Pipeline B accuracy ≥ Pipeline A accuracy
- ✅ False negative rate decreases by ≥ 5%
- ✅ No increase in false positives
- ✅ Agreement rate ≥ 75%
- ✅ No critical bugs
- ✅ Stakeholder approval

**Ideal Targets**:
- 🎯 Pipeline B accuracy +10% over Pipeline A
- 🎯 False negative rate -15%
- 🎯 Agreement rate ≥ 85%
- 🎯 Processing time improvement or neutral

---

### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Pipeline B worse than A** | No-go decision, continue with A |
| **Mixed results** | Iterate on configuration, re-test |
| **Production issues** | Immediate rollback to Pipeline A |
| **Insufficient data** | Extend testing period |
| **User complaints** | Rollback, investigate, fix |

---

## Questions for You

Before starting implementation, please clarify:

1. **Testing Duration**: Is 2-4 weeks acceptable, or do you need faster results?

2. **Test Dataset**: Do you have existing images with known labels, or should we test in production first?

3. **Traffic Split**: For production testing, is 10-20% to new system acceptable?

4. **Rollback Authority**: Who can authorize immediate rollback if issues occur?

5. **Success Criteria**: Are the proposed metrics (accuracy, false negatives, etc.) aligned with your business needs?

---

This plan provides a comprehensive, safe approach to testing the improvements while maintaining production stability. Ready to implement when you give the go-ahead! 🚀


---

## UPDATE: ArcFace Replaces Facenet512 (2025-11-20)

### Why the Change?

**Facenet512 Performance Issues**:
- 10-40x slower than VGG-Face (2-4s vs 50-100ms inference)
- Regular timeouts in production environment
- Only +2-3% accuracy improvement over VGG-Face
- Not production-viable without expensive GPU infrastructure

**ArcFace Advantages**:
- ✅ **State-of-the-art accuracy**: 99.8% LFW (vs VGG-Face 95-97%, Facenet512 97-99%)
- ✅ **Fast inference**: 17ms (comparable to VGG-Face, 100x faster than Facenet512)
- ✅ **Production-ready**: Works excellently on CPU, no timeout issues
- ✅ **Better than both**: Superior to VGG-Face AND Facenet512 in all metrics

### Updated Pipeline Configuration

**Pipeline B (NEW)**:
- Model: ArcFace (2019, InsightFace)
- Detector: RetinaFace (same as VGG-Face)
- Threshold: 0.50 (higher threshold typical for ArcFace)
- Expected performance: 99% accuracy, 0.2-0.4s recognition time

### Implementation Status

✅ **Completed Changes**:
1. Created `ArcFaceSystemProfile` in `app/config/recognition_profiles.py`
2. Updated `ProfileManager` to use 'arcface' instead of 'improved'
3. Updated `test_recognition_service.py` to test ArcFace as Pipeline B
4. Updated `comparison_service.py` to reflect new pipeline name
5. Documentation updated

**Ready for Testing**: System is now configured to A/B test VGG-Face vs ArcFace

### Expected Results

Based on research and benchmarks:

| Metric | VGG-Face (Current) | ArcFace (NEW) | Improvement |
|--------|-------------------|---------------|-------------|
| LFW Accuracy | 95-97% | 99.8% | +2-4% |
| Serbia DB | 96-98% | 97-99% | +1-2% |
| Inference Time | 50-100ms | 17ms | 3-6x faster |
| Recognition Time | 0.3-0.5s | 0.2-0.4s | 20-40% faster |
| Timeouts | 0% | 0% | Stable |

### Next Steps

1. Deploy to testing environment
2. Run 100-500 A/B comparisons over 2-4 weeks
3. Monitor: accuracy, speed, failures, confidence scores
4. Make data-driven decision on migration

See `AB_TESTING_PERFORMANCE_ANALYSIS.md` for detailed analysis.

