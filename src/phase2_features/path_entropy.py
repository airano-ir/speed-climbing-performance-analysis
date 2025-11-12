"""
Path Entropy Calculator for Speed Climbing Analysis
===================================================

Calculates trajectory entropy to measure movement efficiency.
Lower entropy = more direct path = better performance.

Functions:
    - calculate_path_entropy: Main entropy calculation
    - analyze_trajectory_segments: Segment-wise analysis
    - compare_trajectories: Multi-trajectory comparison

References:
    - "Gender-specific biomechanics in speed climbing" (IFSC 2023)
    - Women optimal H: 0.14, Men optimal H: 0.10
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import distance
from scipy.interpolate import interp1d
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_path_entropy(
    com_trajectory: np.ndarray,
    wall_height: float = 15.0,
    num_bins: int = 20,
    method: str = 'shannon'
) -> Dict[str, float]:
    """
    محاسبه آنتروپی مسیر (Path Entropy) از trajectory مرکز جرم.

    آنتروپی کمتر → مسیر مستقیم‌تر → عملکرد بهتر

    Args:
        com_trajectory: Array با shape (N, 2) - مختصات (x, y) مرکز جرم
                        به متر (پس از perspective correction)
        wall_height: ارتفاع دیوار (پیش‌فرض: 15m IFSC standard)
        num_bins: تعداد بازه‌ها برای محاسبه توزیع (پیش‌فرض: 20)
        method: روش محاسبه ('shannon' یا 'sample')

    Returns:
        Dict شامل:
            - 'entropy': آنتروپی مسیر (0 تا ~1)
            - 'avg_deviation': میانگین انحراف از خط مستقیم (متر)
            - 'max_deviation': بیشترین انحراف (متر)
            - 'std_deviation': انحراف معیار
            - 'path_efficiency': نسبت مسیر مستقیم به مسیر واقعی
            - 'lateral_movement_ratio': نسبت حرکت جانبی
            - 'vertical_efficiency': نسبت پیشرفت عمودی

    Raises:
        ValueError: اگر trajectory کمتر از 2 نقطه داشته باشد

    Example:
        >>> trajectory = np.array([[1.5, 0], [1.6, 2.5], [1.4, 5.0], ...])
        >>> result = calculate_path_entropy(trajectory)
        >>> print(f"Entropy: {result['entropy']:.3f}")
        >>> # Optimal: < 0.12, Acceptable: 0.12-0.18, Poor: > 0.18

    References:
        - "Gender-specific biomechanics in speed climbing" (IFSC 2023)
        - Women optimal H: 0.14, Men optimal H: 0.10
    """

    if len(com_trajectory) < 2:
        raise ValueError("Trajectory باید حداقل 2 نقطه داشته باشد")

    # 1. محاسبه خط مستقیم (start → end)
    start_point = com_trajectory[0]
    end_point = com_trajectory[-1]

    # 2. محاسبه انحراف‌ها از خط مستقیم
    deviations = []
    for point in com_trajectory:
        # فاصله عمودی از خط مستقیم
        deviation = _point_to_line_distance(point, start_point, end_point)
        deviations.append(deviation)

    deviations = np.array(deviations)

    # 3. محاسبه آماره‌های پایه
    avg_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    std_deviation = np.std(deviations)

    # 4. Normalize deviations به [0, 1]
    max_dev = max_deviation if max_deviation > 0 else 1e-6
    normalized_devs = deviations / max_dev

    # 5. ساخت هیستوگرام (probability distribution)
    hist, bin_edges = np.histogram(normalized_devs, bins=num_bins, range=(0, 1))

    # Normalize to probabilities
    hist = hist / np.sum(hist)

    # Remove zero bins (log(0) undefined)
    hist = hist[hist > 0]

    # 6. محاسبه Shannon Entropy: H = -Σ p_i * log2(p_i)
    if method == 'shannon':
        entropy = -np.sum(hist * np.log2(hist))
        # Normalize به [0, 1] (max entropy برای num_bins)
        max_entropy = np.log2(num_bins)
        normalized_entropy = entropy / max_entropy
    elif method == 'sample':
        # Sample entropy (alternative method)
        normalized_entropy = _calculate_sample_entropy(normalized_devs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 7. محاسبه path efficiency
    straight_line_distance = np.linalg.norm(end_point - start_point)
    actual_path_length = _compute_path_length(com_trajectory)
    path_efficiency = straight_line_distance / actual_path_length if actual_path_length > 0 else 0

    # 8. محاسبه lateral movement ratio
    lateral_movement = _compute_lateral_movement(com_trajectory)
    lateral_ratio = lateral_movement / actual_path_length if actual_path_length > 0 else 0

    # 9. محاسبه vertical efficiency
    vertical_distance = abs(end_point[1] - start_point[1])
    vertical_efficiency = vertical_distance / actual_path_length if actual_path_length > 0 else 0

    return {
        'entropy': float(normalized_entropy),
        'avg_deviation': float(avg_deviation),
        'max_deviation': float(max_deviation),
        'std_deviation': float(std_deviation),
        'path_efficiency': float(path_efficiency),
        'lateral_movement_ratio': float(lateral_ratio),
        'vertical_efficiency': float(vertical_efficiency),
        'straight_line_distance': float(straight_line_distance),
        'actual_path_length': float(actual_path_length),
        'raw_entropy_bits': float(-np.sum(hist * np.log2(hist))) if method == 'shannon' else 0.0
    }


def _point_to_line_distance(
    point: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray
) -> float:
    """
    محاسبه فاصله عمودی یک نقطه از یک خط (2D).

    Formula: ||(P - A) - ((P - A)·(B - A) / ||B - A||²) * (B - A)||

    Args:
        point: نقطه مورد نظر [x, y]
        line_start: نقطه شروع خط [x, y]
        line_end: نقطه پایان خط [x, y]

    Returns:
        فاصله عمودی (متر)
    """
    line_vec = line_end - line_start
    point_vec = point - line_start

    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-10:  # خط تقریباً نقطه است
        return np.linalg.norm(point_vec)

    line_unitvec = line_vec / line_len
    projection = np.dot(point_vec, line_unitvec)

    # نقطه projection روی خط
    proj_point = line_start + projection * line_unitvec

    # فاصله عمودی
    dist = np.linalg.norm(point - proj_point)
    return dist


def _compute_path_length(trajectory: np.ndarray) -> float:
    """
    محاسبه طول کل مسیر (مجموع Euclidean distances بین نقاط متوالی).

    Args:
        trajectory: Array با shape (N, 2)

    Returns:
        طول مسیر (متر)
    """
    if len(trajectory) < 2:
        return 0.0

    total_length = 0.0
    for i in range(1, len(trajectory)):
        segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
        total_length += segment_length

    return total_length


def _compute_lateral_movement(trajectory: np.ndarray) -> float:
    """
    محاسبه حرکت جانبی (مجموع تغییرات مختصات x).

    Args:
        trajectory: Array با shape (N, 2)

    Returns:
        حرکت جانبی کل (متر)
    """
    if len(trajectory) < 2:
        return 0.0

    lateral = 0.0
    for i in range(1, len(trajectory)):
        lateral += abs(trajectory[i][0] - trajectory[i-1][0])

    return lateral


def _calculate_sample_entropy(signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """
    محاسبه Sample Entropy (روش جایگزین).

    Sample Entropy: معیاری برای پیچیدگی و نامنظمی سیگنال

    Args:
        signal: آرایه یک‌بعدی
        m: طول الگو
        r: threshold tolerance

    Returns:
        Sample entropy (0 تا ~2)
    """
    N = len(signal)

    def _maxdist(xmi, xmj):
        return max([abs(ua - va) for ua, va in zip(xmi, xmj)])

    def _phi(m):
        x = [[signal[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for xmj in x if _maxdist(xmi, xmj) <= r]) / (N - m + 1.0) for xmi in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))

    return abs(_phi(m + 1) - _phi(m))


def analyze_trajectory_segments(
    com_trajectory: np.ndarray,
    num_segments: int = 3,
    segment_heights: Optional[List[float]] = None
) -> Dict[str, List[Dict]]:
    """
    تحلیل trajectory به صورت بخش‌بندی شده (مثلاً: start, middle, finish).

    Args:
        com_trajectory: Array با shape (N, 2)
        num_segments: تعداد بخش‌ها (پیش‌فرض: 3)
        segment_heights: ارتفاع‌های سفارشی برای بخش‌بندی [مثلاً: [0, 5, 10, 15]]

    Returns:
        Dict شامل لیست metrics برای هر segment

    Example:
        >>> result = analyze_trajectory_segments(trajectory, num_segments=3)
        >>> for i, seg in enumerate(result['segments']):
        ...     print(f"Segment {i}: entropy={seg['entropy']:.3f}")
    """
    if segment_heights is None:
        # بخش‌بندی یکنواخت بر اساس ارتفاع
        min_y = com_trajectory[:, 1].min()
        max_y = com_trajectory[:, 1].max()
        segment_heights = np.linspace(min_y, max_y, num_segments + 1)

    segments = []

    for i in range(len(segment_heights) - 1):
        y_start = segment_heights[i]
        y_end = segment_heights[i + 1]

        # انتخاب نقاط در این بخش
        mask = (com_trajectory[:, 1] >= y_start) & (com_trajectory[:, 1] < y_end)
        segment_traj = com_trajectory[mask]

        if len(segment_traj) < 2:
            logger.warning(f"Segment {i} has insufficient points")
            continue

        # محاسبه metrics برای این بخش
        segment_metrics = calculate_path_entropy(segment_traj)
        segment_metrics['segment_id'] = i
        segment_metrics['y_range'] = [float(y_start), float(y_end)]
        segment_metrics['num_points'] = len(segment_traj)

        segments.append(segment_metrics)

    return {
        'segments': segments,
        'num_segments': len(segments)
    }


def compare_trajectories(
    trajectories: List[np.ndarray],
    labels: Optional[List[str]] = None
) -> Dict:
    """
    مقایسه چند trajectory با یکدیگر.

    Args:
        trajectories: لیست از numpy arrays (هر کدام shape (N, 2))
        labels: برچسب‌های اختیاری برای هر trajectory

    Returns:
        Dict شامل metrics مقایسه‌ای

    Example:
        >>> elite_traj = np.array([...])
        >>> amateur_traj = np.array([...])
        >>> comparison = compare_trajectories(
        ...     [elite_traj, amateur_traj],
        ...     labels=['Elite', 'Amateur']
        ... )
        >>> print(comparison['summary'])
    """
    if labels is None:
        labels = [f"Trajectory_{i}" for i in range(len(trajectories))]

    results = []

    for traj, label in zip(trajectories, labels):
        metrics = calculate_path_entropy(traj)
        metrics['label'] = label
        results.append(metrics)

    # Ranking
    sorted_by_entropy = sorted(results, key=lambda x: x['entropy'])

    # Summary statistics
    entropies = [r['entropy'] for r in results]
    efficiencies = [r['path_efficiency'] for r in results]

    summary = {
        'num_trajectories': len(trajectories),
        'best_performer': sorted_by_entropy[0]['label'],
        'worst_performer': sorted_by_entropy[-1]['label'],
        'avg_entropy': float(np.mean(entropies)),
        'std_entropy': float(np.std(entropies)),
        'avg_efficiency': float(np.mean(efficiencies)),
        'entropy_range': [float(min(entropies)), float(max(entropies))]
    }

    return {
        'results': results,
        'ranking': [r['label'] for r in sorted_by_entropy],
        'summary': summary
    }


def interpret_entropy(entropy: float, gender: str = 'female') -> Dict:
    """
    تفسیر مقدار entropy بر اساس استانداردهای IFSC.

    Args:
        entropy: مقدار entropy محاسبه شده
        gender: 'male' یا 'female'

    Returns:
        Dict شامل تفسیر و توصیه‌ها

    Example:
        >>> interpretation = interpret_entropy(0.14, gender='female')
        >>> print(interpretation['rating'])
        >>> print(interpretation['recommendations'])
    """
    # Thresholds (based on IFSC research)
    if gender.lower() == 'female':
        thresholds = {
            'excellent': 0.12,
            'good': 0.14,
            'acceptable': 0.18
        }
        optimal = 0.14
    else:  # male
        thresholds = {
            'excellent': 0.08,
            'good': 0.10,
            'acceptable': 0.15
        }
        optimal = 0.10

    # Rating
    if entropy <= thresholds['excellent']:
        rating = 'excellent'
        description = "Elite level - مسیر بسیار مستقیم و بهینه"
    elif entropy <= thresholds['good']:
        rating = 'good'
        description = "Good - عملکرد مناسب با امکان بهبود"
    elif entropy <= thresholds['acceptable']:
        rating = 'acceptable'
        description = "Acceptable - نیاز به بهبود تکنیک"
    else:
        rating = 'poor'
        description = "Poor - حرکات جانبی زیاد، نیاز به تمرین"

    # Recommendations
    recommendations = []

    if entropy > optimal:
        diff = entropy - optimal
        time_loss = diff * 10  # تقریبی: هر 0.01 → 0.1s
        recommendations.append(
            f"کاهش {diff:.3f} واحد entropy می‌تواند ~{time_loss:.1f} ثانیه بهبود ایجاد کند"
        )

    if rating in ['acceptable', 'poor']:
        recommendations.extend([
            "تمرکز بر push عمودی به جای جانبی",
            "کاهش چرخش لگن در بخش میانی",
            "حفظ momentum و کاهش توقف‌ها",
            "استفاده از dynamic movements"
        ])

    return {
        'entropy': entropy,
        'rating': rating,
        'description': description,
        'optimal_value': optimal,
        'difference_from_optimal': entropy - optimal,
        'recommendations': recommendations
    }


# ==================== Example Usage ====================
if __name__ == "__main__":
    # Simulate COM trajectory (متر)
    print("=== Path Entropy Calculator - Test ===\n")

    # مسیر 1: مستقیم (Optimal)
    straight_trajectory = np.linspace([1.5, 0], [1.6, 15], num=100)

    # مسیر 2: پرانحراف (Poor technique)
    noisy_trajectory = straight_trajectory.copy()
    noisy_trajectory[:, 0] += 0.3 * np.sin(np.linspace(0, 4*np.pi, 100))

    # مسیر 3: متوسط
    moderate_trajectory = straight_trajectory.copy()
    moderate_trajectory[:, 0] += 0.1 * np.sin(np.linspace(0, 2*np.pi, 100))

    # تحلیل مسیر مستقیم
    print("=== Straight Path (Optimal) ===")
    result_optimal = calculate_path_entropy(straight_trajectory)
    for key, value in result_optimal.items():
        print(f"{key:25s}: {value:.4f}")

    interpretation = interpret_entropy(result_optimal['entropy'], gender='female')
    print(f"\nRating: {interpretation['rating']}")
    print(f"Description: {interpretation['description']}")

    # تحلیل مسیر پرانحراف
    print("\n\n=== Noisy Path (Poor Technique) ===")
    result_poor = calculate_path_entropy(noisy_trajectory)
    for key, value in result_poor.items():
        print(f"{key:25s}: {value:.4f}")

    interpretation = interpret_entropy(result_poor['entropy'], gender='female')
    print(f"\nRating: {interpretation['rating']}")
    print(f"Description: {interpretation['description']}")
    print("\nRecommendations:")
    for rec in interpretation['recommendations']:
        print(f"  - {rec}")

    # مقایسه
    print("\n\n=== Trajectory Comparison ===")
    comparison = compare_trajectories(
        [straight_trajectory, moderate_trajectory, noisy_trajectory],
        labels=['Optimal', 'Moderate', 'Poor']
    )

    print(f"Best performer: {comparison['summary']['best_performer']}")
    print(f"Worst performer: {comparison['summary']['worst_performer']}")
    print(f"Average entropy: {comparison['summary']['avg_entropy']:.4f}")
    print(f"\nRanking: {' > '.join(comparison['ranking'])}")

    # تحلیل بخش‌بندی شده
    print("\n\n=== Segment Analysis (Noisy Path) ===")
    segments = analyze_trajectory_segments(noisy_trajectory, num_segments=3)
    for seg in segments['segments']:
        print(f"\nSegment {seg['segment_id']}: Height {seg['y_range'][0]:.1f}m - {seg['y_range'][1]:.1f}m")
        print(f"  Entropy: {seg['entropy']:.4f}")
        print(f"  Efficiency: {seg['path_efficiency']:.4f}")
        print(f"  Points: {seg['num_points']}")
