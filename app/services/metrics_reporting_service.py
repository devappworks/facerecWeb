import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
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
                    f"STRONG RECOMMENDATION: Pipeline B shows {improvement}% accuracy improvement. Consider migration."
                )
            elif improvement > 5:
                recommendations.append(
                    f"Pipeline B shows {improvement}% accuracy improvement. Continue testing."
                )
            elif improvement > 0:
                recommendations.append(
                    f"Pipeline B shows modest {improvement}% improvement. Collect more data."
                )
            elif improvement < -5:
                recommendations.append(
                    f"WARNING: Pipeline B accuracy is {abs(improvement)}% worse. Review configuration."
                )

        # Check success rate
        status = summary.get("status_breakdown", {})
        only_b_succeeded = status.get("only_b_succeeded", {}).get("percentage", 0)
        only_a_succeeded = status.get("only_a_succeeded", {}).get("percentage", 0)

        if only_b_succeeded > only_a_succeeded + 10:
            recommendations.append(
                f"Pipeline B finds faces that Pipeline A misses ({only_b_succeeded}% vs {only_a_succeeded}%). Good sign!"
            )
        elif only_a_succeeded > only_b_succeeded + 10:
            recommendations.append(
                f"Pipeline A finds more faces than Pipeline B ({only_a_succeeded}% vs {only_b_succeeded}%). Investigate."
            )

        # Check agreement rate
        agreement = summary.get("agreement", {})
        agreement_rate = agreement.get("agreement_rate", 0)

        if agreement_rate > 90:
            recommendations.append(
                f"High agreement rate ({agreement_rate}%). Pipelines are consistent."
            )
        elif agreement_rate < 70:
            recommendations.append(
                f"Low agreement rate ({agreement_rate}%). Review disagreement cases."
            )

        # Check performance
        performance = summary.get("performance", {})
        time_diff = performance.get("avg_time_difference_ms")

        if time_diff and time_diff < -100:
            recommendations.append(
                f"Pipeline B is {abs(time_diff):.0f}ms faster on average."
            )
        elif time_diff and time_diff > 500:
            recommendations.append(
                f"Pipeline B is {time_diff:.0f}ms slower. May need optimization."
            )

        if not recommendations:
            recommendations.append("Continue collecting data for conclusive recommendations.")

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
        import json
        comparison = None

        for file_path in glob.glob(f"storage/comparisons/*.json"):
            try:
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
                report_lines.append("  Both pipelines agree")
            else:
                report_lines.append("  Pipelines disagree!")

            if metrics.get("confidence_difference"):
                diff = metrics["confidence_difference"]
                symbol = "+" if diff > 0 else ""
                report_lines.append(f"  Confidence Difference: {symbol}{diff}%")

            if metrics.get("faster_pipeline"):
                faster = "Pipeline B" if metrics["faster_pipeline"] == "pipeline_b" else "Pipeline A"
                time_diff = abs(metrics.get("processing_time_difference", 0))
                report_lines.append(f"  Faster: {faster} by {time_diff*1000:.0f}ms")

        elif metrics["only_b_succeeded"]:
            report_lines.append("  Only Pipeline B found a face")

        elif metrics["only_a_succeeded"]:
            report_lines.append("  Only Pipeline A found a face")

        else:
            report_lines.append("  Both pipelines failed")

        # Accuracy (if ground truth)
        if metrics.get("accuracy"):
            acc = metrics["accuracy"]
            report_lines.extend([
                "",
                "ACCURACY:",
                f"  Pipeline A: {'Correct' if acc['pipeline_a_correct'] else 'Wrong'}",
                f"  Pipeline B: {'Correct' if acc['pipeline_b_correct'] else 'Wrong'}",
                f"  Winner: {acc['winner'].upper()}"
            ])

        report_lines.extend([
            "",
            "="*60
        ])

        return "\n".join(report_lines)
