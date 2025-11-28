"""
Feedback Generator for Speed Climbing Analysis.

Generates human-readable, personalized feedback in Persian and English.

UPDATED: Only uses camera-independent features (angles, ratios, sync).
Removed efficiency features that are artifacts of camera following athlete.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .fuzzy_engine import FuzzyFeedbackEngine, FuzzyLevel, PerformanceCategory
from .baseline import BaselineStatistics


class Language(Enum):
    PERSIAN = "fa"
    ENGLISH = "en"


@dataclass
class Feedback:
    """Complete feedback package for an athlete."""
    overall_score: float
    overall_level: str
    overall_summary: str

    strengths: List[Dict[str, str]]
    improvements: List[Dict[str, str]]
    recommendations: List[Dict[str, str]]

    category_scores: Dict[str, float]
    category_details: Dict[str, Dict]

    comparison_text: str
    training_tips: List[str]

    raw_features: Dict[str, float] = field(default_factory=dict)


class FeedbackGenerator:
    """
    Generates personalized feedback from performance analysis.

    Supports bilingual output (Persian/English).

    NOTE: Only uses camera-independent features.
    """

    # Feature descriptions for feedback - ONLY VALID FEATURES
    FEATURE_INFO = {
        # Coordination features
        'freq_limb_sync_ratio': {
            'name_en': 'Hand-Foot Coordination',
            'name_fa': 'هماهنگی دست و پا',
            'good_en': 'Excellent limb coordination',
            'good_fa': 'هماهنگی عالی اندام‌ها',
            'bad_en': 'Hand and foot movements need better sync',
            'bad_fa': 'هماهنگی دست و پا نیاز به بهبود دارد',
            'tip_en': 'Practice coordinated climbing drills',
            'tip_fa': 'تمرین تمرینات صعود هماهنگ',
        },
        'freq_hand_movement_amplitude': {
            'name_en': 'Hand Movement Range',
            'name_fa': 'دامنه حرکت دست',
            'good_en': 'Good hand movement amplitude',
            'good_fa': 'دامنه حرکت دست مناسب',
            'bad_en': 'Hand movements are too small or too large',
            'bad_fa': 'حرکات دست خیلی کوچک یا خیلی بزرگ است',
            'tip_en': 'Practice controlled reach movements',
            'tip_fa': 'تمرین حرکات کنترل‌شده دست',
        },
        'freq_foot_movement_amplitude': {
            'name_en': 'Foot Movement Range',
            'name_fa': 'دامنه حرکت پا',
            'good_en': 'Good foot movement amplitude',
            'good_fa': 'دامنه حرکت پای مناسب',
            'bad_en': 'Foot movements need adjustment',
            'bad_fa': 'حرکات پا نیاز به تنظیم دارد',
            'tip_en': 'Focus on precise foot placements',
            'tip_fa': 'تمرکز بر قرار دادن دقیق پا',
        },

        # Leg technique features
        'post_avg_knee_angle': {
            'name_en': 'Knee Angle',
            'name_fa': 'زاویه زانو',
            'good_en': 'Good knee bend for power',
            'good_fa': 'خم شدن مناسب زانو برای قدرت',
            'bad_en': 'Knee angle needs adjustment',
            'bad_fa': 'زاویه زانو نیاز به تنظیم دارد',
            'tip_en': 'Practice driving up with bent knees',
            'tip_fa': 'تمرین بلند شدن با زانوهای خمیده',
        },
        'post_knee_angle_std': {
            'name_en': 'Knee Angle Consistency',
            'name_fa': 'یکنواختی زاویه زانو',
            'good_en': 'Consistent knee technique',
            'good_fa': 'تکنیک زانوی یکنواخت',
            'bad_en': 'Knee angle varies too much',
            'bad_fa': 'زاویه زانو تغییرات زیادی دارد',
            'tip_en': 'Focus on consistent leg drive',
            'tip_fa': 'تمرکز بر فشار یکنواخت پا',
        },

        # Arm technique features
        'post_avg_elbow_angle': {
            'name_en': 'Elbow Angle',
            'name_fa': 'زاویه آرنج',
            'good_en': 'Efficient arm extension',
            'good_fa': 'کشش کارآمد بازو',
            'bad_en': 'Arms are too bent or too straight',
            'bad_fa': 'بازوها خیلی خمیده یا خیلی صاف هستند',
            'tip_en': 'Keep arms slightly bent, use legs for power',
            'tip_fa': 'بازوها را کمی خمیده نگه دارید، از پاها برای قدرت استفاده کنید',
        },
        'post_elbow_angle_std': {
            'name_en': 'Arm Technique Consistency',
            'name_fa': 'یکنواختی تکنیک دست',
            'good_en': 'Consistent arm technique',
            'good_fa': 'تکنیک دست یکنواخت',
            'bad_en': 'Arm technique varies too much',
            'bad_fa': 'تکنیک دست تغییرات زیادی دارد',
            'tip_en': 'Practice smooth arm transitions',
            'tip_fa': 'تمرین انتقال‌های روان دست',
        },

        # Body position features
        'post_avg_body_lean': {
            'name_en': 'Body Angle',
            'name_fa': 'زاویه بدن',
            'good_en': 'Optimal body position close to wall',
            'good_fa': 'وضعیت بهینه بدن نزدیک دیوار',
            'bad_en': 'Body leans too far from wall',
            'bad_fa': 'بدن خیلی از دیوار فاصله دارد',
            'tip_en': 'Stay close to the wall, hips in',
            'tip_fa': 'نزدیک دیوار بمانید، لگن به داخل',
        },
        'post_body_lean_std': {
            'name_en': 'Body Position Stability',
            'name_fa': 'ثبات وضعیت بدن',
            'good_en': 'Consistent body position throughout climb',
            'good_fa': 'وضعیت ثابت بدن در طول صعود',
            'bad_en': 'Body position varies too much',
            'bad_fa': 'وضعیت بدن تغییرات زیادی دارد',
            'tip_en': 'Focus on controlled core movements',
            'tip_fa': 'تمرکز بر حرکات کنترل‌شده مرکزی',
        },
        'post_hip_width_ratio': {
            'name_en': 'Hip Position',
            'name_fa': 'وضعیت لگن',
            'good_en': 'Good hip positioning for balance',
            'good_fa': 'وضعیت مناسب لگن برای تعادل',
            'bad_en': 'Hip position needs adjustment',
            'bad_fa': 'وضعیت لگن نیاز به تنظیم دارد',
            'tip_en': 'Keep hips centered and close to wall',
            'tip_fa': 'لگن را مرکز و نزدیک دیوار نگه دارید',
        },

        # Reach features
        'post_avg_reach_ratio': {
            'name_en': 'Average Reach',
            'name_fa': 'دسترسی میانگین',
            'good_en': 'Good use of reach relative to body',
            'good_fa': 'استفاده خوب از دسترسی نسبت به بدن',
            'bad_en': 'Reach could be more efficient',
            'bad_fa': 'دسترسی می‌تواند کارآمدتر باشد',
            'tip_en': 'Extend fully before moving feet',
            'tip_fa': 'قبل از حرکت پا، کاملاً کشش دهید',
        },
        'post_max_reach_ratio': {
            'name_en': 'Maximum Reach',
            'name_fa': 'حداکثر دسترسی',
            'good_en': 'Excellent maximum extension',
            'good_fa': 'کشش حداکثری عالی',
            'bad_en': 'Not using full reach potential',
            'bad_fa': 'از پتانسیل کامل دسترسی استفاده نمی‌شود',
            'tip_en': 'Practice dynamic reaches',
            'tip_fa': 'تمرین دسترسی‌های پویا',
        },
    }

    # Level descriptions
    LEVEL_TEXT = {
        FuzzyLevel.VERY_HIGH: {
            'en': 'Elite',
            'fa': 'نخبه',
            'desc_en': 'Professional level technique',
            'desc_fa': 'تکنیک سطح حرفه‌ای',
        },
        FuzzyLevel.HIGH: {
            'en': 'Advanced',
            'fa': 'پیشرفته',
            'desc_en': 'Strong technique, approaching elite',
            'desc_fa': 'تکنیک قوی، نزدیک به سطح نخبه',
        },
        FuzzyLevel.MEDIUM: {
            'en': 'Intermediate',
            'fa': 'متوسط',
            'desc_en': 'Solid foundation with room to grow',
            'desc_fa': 'پایه محکم با فضا برای رشد',
        },
        FuzzyLevel.LOW: {
            'en': 'Developing',
            'fa': 'در حال رشد',
            'desc_en': 'Building skills, keep practicing',
            'desc_fa': 'در حال ساختن مهارت‌ها، به تمرین ادامه دهید',
        },
        FuzzyLevel.VERY_LOW: {
            'en': 'Beginner',
            'fa': 'مبتدی',
            'desc_en': 'Early stage, focus on fundamentals',
            'desc_fa': 'مرحله اولیه، بر اصول تمرکز کنید',
        },
    }

    def __init__(
        self,
        language: Language = Language.PERSIAN,
        baseline: Optional[BaselineStatistics] = None
    ):
        """
        Initialize feedback generator.

        Args:
            language: Output language (Persian or English)
            baseline: Baseline statistics for comparison
        """
        self.language = language
        self.fuzzy_engine = FuzzyFeedbackEngine(baseline)
        self.baseline = baseline or BaselineStatistics()

    def generate(self, features: Dict[str, float]) -> Feedback:
        """
        Generate complete feedback from features.

        Args:
            features: Dict of feature_name -> value

        Returns:
            Feedback object with all analysis
        """
        # Get overall score
        overall_score, overall_level = self.fuzzy_engine.get_overall_score(features)

        # Evaluate all categories
        categories = self.fuzzy_engine.evaluate_all(features)

        # Generate text
        lang = 'fa' if self.language == Language.PERSIAN else 'en'

        # Overall summary
        level_info = self.LEVEL_TEXT[overall_level]
        overall_summary = self._format_overall_summary(overall_score, level_info, lang)

        # Collect strengths and improvements
        strengths = self._collect_strengths(categories, features, lang)
        improvements = self._collect_improvements(categories, features, lang)
        recommendations = self._generate_recommendations(categories, features, lang)

        # Category scores and details
        category_scores = {name: cat.score for name, cat in categories.items()}
        category_details = self._format_category_details(categories, lang)

        # Comparison text
        comparison_text = self._generate_comparison_text(overall_score, lang)

        # Training tips
        training_tips = self._generate_training_tips(improvements, lang)

        return Feedback(
            overall_score=overall_score,
            overall_level=level_info[lang],
            overall_summary=overall_summary,
            strengths=strengths,
            improvements=improvements,
            recommendations=recommendations,
            category_scores=category_scores,
            category_details=category_details,
            comparison_text=comparison_text,
            training_tips=training_tips,
            raw_features=features,
        )

    def _format_overall_summary(self, score: float, level_info: Dict, lang: str) -> str:
        """Format overall performance summary."""
        if lang == 'fa':
            return (
                f"امتیاز کلی تکنیک: {score:.0f} از ۱۰۰\n"
                f"سطح: {level_info['fa']}\n"
                f"{level_info['desc_fa']}"
            )
        else:
            return (
                f"Overall Technique Score: {score:.0f}/100\n"
                f"Level: {level_info['en']}\n"
                f"{level_info['desc_en']}"
            )

    def _collect_strengths(
        self,
        categories: Dict[str, PerformanceCategory],
        features: Dict[str, float],
        lang: str
    ) -> List[Dict[str, str]]:
        """Collect strength points from analysis."""
        strengths = []

        for cat_name, cat in categories.items():
            # Category-level strength
            if cat.score >= 70:
                strengths.append({
                    'category': cat.name_fa if lang == 'fa' else cat.name,
                    'text': self._get_category_strength_text(cat_name, lang),
                    'score': f"{cat.score:.0f}",
                })

            # Feature-level strengths
            for feat_name in cat.strengths:
                if feat_name in self.FEATURE_INFO:
                    info = self.FEATURE_INFO[feat_name]
                    strengths.append({
                        'category': cat.name_fa if lang == 'fa' else cat.name,
                        'text': info[f'good_{lang}'],
                        'feature': info[f'name_{lang}'],
                    })

        return strengths[:5]  # Top 5 strengths

    def _collect_improvements(
        self,
        categories: Dict[str, PerformanceCategory],
        features: Dict[str, float],
        lang: str
    ) -> List[Dict[str, str]]:
        """Collect areas for improvement."""
        improvements = []
        seen_features = set()

        for cat_name, cat in categories.items():
            # Feature-level weaknesses
            for feat_name in cat.weaknesses:
                if feat_name in self.FEATURE_INFO and feat_name not in seen_features:
                    seen_features.add(feat_name)
                    info = self.FEATURE_INFO[feat_name]
                    improvements.append({
                        'category': cat.name_fa if lang == 'fa' else cat.name,
                        'text': info[f'bad_{lang}'],
                        'feature': info[f'name_{lang}'],
                        'priority': 'high' if cat.score < 40 else 'medium',
                    })

        # Sort by priority
        improvements.sort(key=lambda x: 0 if x['priority'] == 'high' else 1)
        return improvements[:5]  # Top 5 improvements

    def _generate_recommendations(
        self,
        categories: Dict[str, PerformanceCategory],
        features: Dict[str, float],
        lang: str
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        seen_features = set()

        # Find weakest category
        sorted_cats = sorted(categories.items(), key=lambda x: x[1].score)

        for cat_name, cat in sorted_cats[:2]:  # Focus on 2 weakest
            for feat_name in cat.weaknesses:
                if feat_name in self.FEATURE_INFO and feat_name not in seen_features:
                    seen_features.add(feat_name)
                    info = self.FEATURE_INFO[feat_name]
                    recommendations.append({
                        'area': info[f'name_{lang}'],
                        'action': info[f'tip_{lang}'],
                        'priority': 'high' if cat.score < 40 else 'medium',
                    })

        return recommendations[:4]  # Top 4 recommendations

    def _get_category_strength_text(self, cat_name: str, lang: str) -> str:
        """Get strength text for a category."""
        texts = {
            'coordination': {
                'fa': 'هماهنگی اندام‌ها بسیار خوب است',
                'en': 'Excellent limb coordination',
            },
            'leg_technique': {
                'fa': 'تکنیک پا در سطح بالایی است',
                'en': 'Strong leg technique',
            },
            'arm_technique': {
                'fa': 'تکنیک دست مناسب است',
                'en': 'Good arm technique',
            },
            'body_position': {
                'fa': 'وضعیت بدن بهینه است',
                'en': 'Optimal body positioning',
            },
            'reach': {
                'fa': 'استفاده خوب از دسترسی',
                'en': 'Good use of reach',
            },
        }
        return texts.get(cat_name, {}).get(lang, '')

    def _format_category_details(
        self,
        categories: Dict[str, PerformanceCategory],
        lang: str
    ) -> Dict[str, Dict]:
        """Format detailed category information."""
        details = {}

        for cat_name, cat in categories.items():
            level_info = self.LEVEL_TEXT[cat.level]
            details[cat_name] = {
                'name': cat.name_fa if lang == 'fa' else cat.name,
                'score': cat.score,
                'level': level_info[lang],
                'confidence': cat.confidence,
                'strengths_count': len(cat.strengths),
                'weaknesses_count': len(cat.weaknesses),
            }

        return details

    def _generate_comparison_text(self, score: float, lang: str) -> str:
        """Generate comparison text against professional athletes."""
        percentile = min(99, max(1, score))

        if lang == 'fa':
            if percentile >= 80:
                return f"تکنیک شما در سطح {percentile:.0f}٪ ورزشکاران حرفه‌ای است. عالی!"
            elif percentile >= 60:
                return f"تکنیک شما بهتر از {percentile:.0f}٪ ورزشکاران در دیتاست ما است."
            elif percentile >= 40:
                return f"تکنیک شما در محدوده متوسط قرار دارد ({percentile:.0f}٪)."
            else:
                return f"فضای زیادی برای بهبود تکنیک دارید. با تمرین منظم پیشرفت خواهید کرد."
        else:
            if percentile >= 80:
                return f"Your technique is at the {percentile:.0f}th percentile of pro athletes. Excellent!"
            elif percentile >= 60:
                return f"Your technique is better than {percentile:.0f}% of athletes in our dataset."
            elif percentile >= 40:
                return f"Your technique is in the average range ({percentile:.0f}th percentile)."
            else:
                return f"Lots of room to improve technique. Regular practice will help."

    def _generate_training_tips(
        self,
        improvements: List[Dict],
        lang: str
    ) -> List[str]:
        """Generate training tips based on improvements needed."""
        tips = []

        # Generic tips based on weaknesses
        for imp in improvements[:3]:
            feat_name = None
            for name, info in self.FEATURE_INFO.items():
                if info.get(f'name_{lang}') == imp.get('feature'):
                    feat_name = name
                    break

            if feat_name and feat_name in self.FEATURE_INFO:
                tips.append(self.FEATURE_INFO[feat_name][f'tip_{lang}'])

        # Add general tips if needed
        if lang == 'fa':
            general_tips = [
                "ویدیو از صعود خود بگیرید و تکنیک را تحلیل کنید",
                "روی یک جنبه تکنیک در هر جلسه تمرینی تمرکز کنید",
                "قبل از تمرین سرعت، تکنیک را کامل کنید",
            ]
        else:
            general_tips = [
                "Record and analyze your technique",
                "Focus on one technique aspect per training session",
                "Perfect technique before working on speed",
            ]

        while len(tips) < 3:
            if general_tips:
                tips.append(general_tips.pop(0))
            else:
                break

        return tips

    def format_report(self, feedback: Feedback) -> str:
        """
        Format feedback as a readable text report.

        Returns formatted string for display.
        """
        lang = 'fa' if self.language == Language.PERSIAN else 'en'

        if lang == 'fa':
            return self._format_report_persian(feedback)
        else:
            return self._format_report_english(feedback)

    def _format_report_persian(self, fb: Feedback) -> str:
        """Format report in Persian."""
        lines = [
            "=" * 50,
            "📊 گزارش تحلیل تکنیک صخره‌نوردی سرعت",
            "=" * 50,
            "",
            fb.overall_summary,
            "",
            "─" * 50,
            "",
        ]

        # Strengths
        if fb.strengths:
            lines.append("💪 نقاط قوت تکنیک:")
            for s in fb.strengths:
                lines.append(f"  ✓ {s['text']}")
            lines.append("")

        # Improvements
        if fb.improvements:
            lines.append("⚠️ فرصت‌های بهبود:")
            for imp in fb.improvements:
                priority = "🔴" if imp.get('priority') == 'high' else "🟡"
                lines.append(f"  {priority} {imp['text']}")
            lines.append("")

        # Category scores
        lines.append("📈 امتیاز دسته‌ها:")
        for cat_name, details in fb.category_details.items():
            bar = self._score_bar(details['score'])
            lines.append(f"  {details['name']}: {bar} {details['score']:.0f}")
        lines.append("")

        # Recommendations
        if fb.recommendations:
            lines.append("🎯 توصیه‌های تمرینی:")
            for i, rec in enumerate(fb.recommendations, 1):
                lines.append(f"  {i}. {rec['action']}")
            lines.append("")

        # Comparison
        lines.append("📊 مقایسه با حرفه‌ای‌ها:")
        lines.append(f"  {fb.comparison_text}")
        lines.append("")

        # Note about limitations
        lines.append("─" * 50)
        lines.append("📝 توجه: این تحلیل بر اساس زوایای بدن و هماهنگی است.")
        lines.append("   سرعت واقعی صعود به دلیل حرکت دوربین قابل اندازه‌گیری نیست.")

        lines.append("=" * 50)

        return "\n".join(lines)

    def _format_report_english(self, fb: Feedback) -> str:
        """Format report in English."""
        lines = [
            "=" * 50,
            "📊 Speed Climbing Technique Analysis Report",
            "=" * 50,
            "",
            fb.overall_summary,
            "",
            "─" * 50,
            "",
        ]

        # Strengths
        if fb.strengths:
            lines.append("💪 Technique Strengths:")
            for s in fb.strengths:
                lines.append(f"  ✓ {s['text']}")
            lines.append("")

        # Improvements
        if fb.improvements:
            lines.append("⚠️ Areas for Improvement:")
            for imp in fb.improvements:
                priority = "🔴" if imp.get('priority') == 'high' else "🟡"
                lines.append(f"  {priority} {imp['text']}")
            lines.append("")

        # Category scores
        lines.append("📈 Category Scores:")
        for cat_name, details in fb.category_details.items():
            bar = self._score_bar(details['score'])
            lines.append(f"  {details['name']}: {bar} {details['score']:.0f}")
        lines.append("")

        # Recommendations
        if fb.recommendations:
            lines.append("🎯 Training Recommendations:")
            for i, rec in enumerate(fb.recommendations, 1):
                lines.append(f"  {i}. {rec['action']}")
            lines.append("")

        # Comparison
        lines.append("📊 Comparison with Professionals:")
        lines.append(f"  {fb.comparison_text}")
        lines.append("")

        # Note about limitations
        lines.append("─" * 50)
        lines.append("📝 Note: This analysis is based on body angles and coordination.")
        lines.append("   Actual climbing speed cannot be measured due to camera motion.")

        lines.append("=" * 50)

        return "\n".join(lines)

    def _score_bar(self, score: float, width: int = 10) -> str:
        """Create a visual score bar."""
        filled = int(score / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty
