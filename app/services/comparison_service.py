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
            result_a: Result from pipeline A (VGG-Face - current system)
            result_b: Result from pipeline B (ArcFace - state-of-the-art system)
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
                "name": "arcface_system",
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
                log_msg += f"Both agree: {comparison['pipeline_a']['person']}"
            else:
                log_msg += f"Disagree: A={comparison['pipeline_a']['person']}, B={comparison['pipeline_b']['person']}"

            if metrics["confidence_difference"]:
                log_msg += f" | Conf diff: {metrics['confidence_difference']:+.1f}%"

        elif metrics["only_b_succeeded"]:
            log_msg += f"Only B succeeded: {comparison['pipeline_b']['person']}"

        elif metrics["only_a_succeeded"]:
            log_msg += f"Only A succeeded: {comparison['pipeline_a']['person']}"

        else:
            log_msg += "Both failed"

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
