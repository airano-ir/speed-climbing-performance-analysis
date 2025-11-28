"""
Feedback Generator for Speed Climbing Analysis.

Generates human-readable, personalized feedback in Persian and English.
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
    """

    # Feature descriptions for feedback
    FEATURE_INFO = {
        'freq_hand_frequency_hz': {
            'name_en': 'Hand Movement Speed',
            'name_fa': 'سرعت حرکت دست',
            'good_en': 'Good hand movement rhythm',
            'good_fa': 'ریتم خوب حرکت دست',
            'bad_en': 'Hand movements could be more rhythmic',
            'bad_fa': 'حرکات دست می‌تواند ریتمیک‌تر باشد',
            'tip_en': 'Practice quick, consistent hand placements',
            'tip_fa': 'تمرین قرار دادن سریع و یکنواخت دست‌ها',
        },
        'freq_foot_frequency_hz': {
            'name_en': 'Foot Movement Speed',
            'name_fa': 'سرعت حرکت پا',
            'good_en': 'Efficient foot work',
            'good_fa': 'کار پای کارآمد',
            'bad_en': 'Foot movements need more speed',
            'bad_fa': 'حرکات پا نیاز به سرعت بیشتر دارد',
            'tip_en': 'Focus on quick foot placements without looking',
            'tip_fa': 'تمرکز بر قرار دادن سریع پا بدون نگاه کردن',
        },
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
        'freq_movement_regularity': {
            'name_en': 'Movement Rhythm',
            'name_fa': 'ریتم حرکت',
            'good_en': 'Consistent climbing rhythm',
            'good_fa': 'ریتم صعود یکنواخت',
            'bad_en': 'Rhythm varies too much during climb',
            'bad_fa': 'ریتم در طول صعود تغییرات زیادی دارد',
            'tip_en': 'Use a metronome while training',
            'tip_fa': 'استفاده از مترونوم در تمرین',
        },
        'eff_path_straightness': {
            'name_en': 'Path Efficiency',
            'name_fa': 'کارایی مسیر',
            'good_en': 'Direct, efficient climbing path',
            'good_fa': 'مسیر صعود مستقیم و کارآمد',
            'bad_en': 'Climbing path is not direct enough',
            'bad_fa': 'مسیر صعود به اندازه کافی مستقیم نیست',
            'tip_en': 'Visualize the shortest path before starting',
            'tip_fa': 'کوتاه‌ترین مسیر را قبل از شروع تجسم کنید',
        },
        'eff_lateral_movement_ratio': {
            'name_en': 'Lateral Movement',
            'name_fa': 'حرکات جانبی',
            'good_en': 'Minimal unnecessary sideways movement',
            'good_fa': 'حداقل حرکات جانبی غیرضروری',
            'bad_en': 'Too much sideways movement',
            'bad_fa': 'حرکات جانبی بیش از حد',
            'tip_en': 'Focus on vertical progression',
            'tip_fa': 'تمرکز بر پیشرفت عمودی',
        },
        'eff_com_stability_index': {
            'name_en': 'Center of Mass Stability',
            'name_fa': 'ثبات مرکز ثقل',
            'good_en': 'Stable center of gravity',
            'good_fa': 'مرکز ثقل پایدار',
            'bad_en': 'Body center moves excessively',
            'bad_fa': 'مرکز بدن بیش از حد حرکت می‌کند',
            'tip_en': 'Keep hips close to wall',
            'tip_fa': 'لگن را نزدیک دیوار نگه دارید',
        },
        'eff_movement_smoothness': {
            'name_en': 'Movement Smoothness',
            'name_fa': 'روانی حرکت',
            'good_en': 'Smooth, fluid movements',
            'good_fa': 'حرکات روان و سیال',
            'bad_en': 'Movements are jerky',
            'bad_fa': 'حرکات تکان‌دهنده هستند',
            'tip_en': 'Practice slow, controlled climbing',
            'tip_fa': 'تمرین صعود آهسته و کنترل‌شده',
        },
        'post_avg_knee_angle': {
            'name_en': 'Knee Position',
            'name_fa': 'وضعیت زانو',
            'good_en': 'Good knee bend for power',
            'good_fa': 'خم شدن مناسب زانو برای قدرت',
            'bad_en': 'Knee angle needs adjustment',
            'bad_fa': 'زاویه زانو نیاز به تنظیم دارد',
            'tip_en': 'Practice driving up with bent knees',
            'tip_fa': 'تمرین بلند شدن با زانوهای خمیده',
        },
        'post_avg_elbow_angle': {
            'name_en': 'Arm Position',
            'name_fa': 'وضعیت بازو',
            'good_en': 'Efficient arm extension',
            'good_fa': 'کشش کارآمد بازو',
            'bad_en': 'Arms are too bent or too straight',
            'bad_fa': 'بازوها خیلی خمیده یا خیلی صاف هستند',
            'tip_en': 'Keep arms slightly bent, use legs for power',
            'tip_fa': 'بازوها را کمی خمیده نگه دارید، از پاها برای قدرت استفاده کنید',
        },
        'post_avg_body_lean': {
            'name_en': 'Body Angle',
            'name_fa': 'زاویه بدن',
            'good_en': 'Optimal body position',
            'good_fa': 'وضعیت بهینه بدن',
            'bad_en': 'Body leans too far from wall',
            'bad_fa': 'بدن خیلی از دیوار فاصله دارد',
            'tip_en': 'Stay close to the wall',
            'tip_fa': 'نزدیک دیوار بمانید',
        },
        'post_body_lean_std': {
            'name_en': 'Body Stability',
            'name_fa': 'ثبات بدن',
            'good_en': 'Consistent body position',
            'good_fa': 'وضعیت ثابت بدن',
            'bad_en': 'Body position varies too much',
            'bad_fa': 'وضعیت بدن تغییرات زیادی دارد',
            'tip_en': 'Focus on controlled movements',
            'tip_fa': 'تمرکز بر حرکات کنترل‌شده',
        },
    }

    # Level descriptions
    LEVEL_TEXT = {
        FuzzyLevel.VERY_HIGH: {
            'en': 'Elite',
            'fa': 'نخبه',
            'desc_en': 'Professional level performance',
            'desc_fa': 'عملکرد سطح حرفه‌ای',
        },
        FuzzyLevel.HIGH: {
            'en': 'Advanced',
            'fa': 'پیشرفته',
            'desc_en': 'Strong performance, approaching elite',
            'desc_fa': 'عملکرد قوی، نزدیک به سطح نخبه',
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
                f"امتیاز کلی شما: {score:.0f} از ۱۰۰\n"
                f"سطح: {level_info['fa']}\n"
                f"{level_info['desc_fa']}"
            )
        else:
            return (
                f"Overall Score: {score:.0f}/100\n"
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
            'rhythm': {
                'fa': 'ریتم و هماهنگی حرکات بسیار خوب است',
                'en': 'Excellent rhythm and coordination',
            },
            'efficiency': {
                'fa': 'کارایی حرکت در سطح بالایی است',
                'en': 'High movement efficiency',
            },
            'stability': {
                'fa': 'تعادل و ثبات عالی',
                'en': 'Excellent balance and stability',
            },
            'posture': {
                'fa': 'وضعیت بدن مناسب',
                'en': 'Good body posture',
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
                return f"عملکرد شما در سطح {percentile:.0f}٪ ورزشکاران حرفه‌ای است. عالی!"
            elif percentile >= 60:
                return f"شما بهتر از {percentile:.0f}٪ ورزشکاران در دیتاست ما عمل کرده‌اید."
            elif percentile >= 40:
                return f"شما در محدوده متوسط قرار دارید ({percentile:.0f}٪)."
            else:
                return f"فضای زیادی برای پیشرفت دارید. با تمرین منظم می‌توانید بهبود یابید."
        else:
            if percentile >= 80:
                return f"Your performance is at the {percentile:.0f}th percentile of pro athletes. Excellent!"
            elif percentile >= 60:
                return f"You performed better than {percentile:.0f}% of athletes in our dataset."
            elif percentile >= 40:
                return f"You are in the average range ({percentile:.0f}th percentile)."
            else:
                return f"Lots of room to grow. Regular practice will help you improve."

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
                "ویدیو از صعود خود بگیرید و تحلیل کنید",
                "روی یک جنبه در هر جلسه تمرینی تمرکز کنید",
                "قبل از تمرین سرعت، تکنیک را کامل کنید",
            ]
        else:
            general_tips = [
                "Record and analyze your climbs",
                "Focus on one aspect per training session",
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
            "📊 گزارش تحلیل عملکرد صخره‌نوردی سرعت",
            "=" * 50,
            "",
            fb.overall_summary,
            "",
            "─" * 50,
            "",
        ]

        # Strengths
        if fb.strengths:
            lines.append("💪 نقاط قوت:")
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

        lines.append("=" * 50)

        return "\n".join(lines)

    def _format_report_english(self, fb: Feedback) -> str:
        """Format report in English."""
        lines = [
            "=" * 50,
            "📊 Speed Climbing Performance Analysis Report",
            "=" * 50,
            "",
            fb.overall_summary,
            "",
            "─" * 50,
            "",
        ]

        # Strengths
        if fb.strengths:
            lines.append("💪 Strengths:")
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

        lines.append("=" * 50)

        return "\n".join(lines)

    def _score_bar(self, score: float, width: int = 10) -> str:
        """Create a visual score bar."""
        filled = int(score / 100 * width)
        empty = width - filled
        return "█" * filled + "░" * empty
