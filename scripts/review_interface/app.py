"""
Manual Review Interface - Main Streamlit App
============================================
رابط کاربری بررسی دستی - اپلیکیشن اصلی Streamlit

A bilingual (English/Persian) interface for manually reviewing and correcting
race detection errors in speed climbing videos.

Version: 1.0
Date: 2025-11-16
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.review_interface.config import ConfigManager
from scripts.review_interface.progress import ProgressTracker, RaceReviewStatus
from scripts.review_interface.metadata_manager import MetadataManager
from scripts.review_interface.video_player import VideoPlayer
from scripts.review_interface.validators import RaceValidator
from scripts.review_interface.video_library import VideoLibrary
from scripts.review_interface.video_extraction import VideoExtractor
from scripts.review_interface.bulk_operations import BulkOperationsManager
from scripts.review_interface.phase_manager import PhaseManager, Phase


# =============================================================================
# LANGUAGE SUPPORT / پشتیبانی زبان
# =============================================================================

TRANSLATIONS = {
    'en': {
        'page_title': '🏔️ Speed Climbing - Manual Race Review Interface',
        'subtitle': 'Fix race detection errors for suspicious races',
        'sidebar_stats': '📊 Progress Statistics',
        'total_races': 'Total Races',
        'completed': 'Completed',
        'pending': 'Pending',
        'skipped': 'Skipped',
        'critical': 'Critical',
        'filter_races': '🎯 Filter Races',
        'competition': 'Competition',
        'priority': 'Priority',
        'status': 'Status',
        'all': 'All',
        'pending_only': 'Pending Only',
        'race_review': '🏁 Race Review',
        'showing_races': '📋 Showing {count} races (filtered from {total} total)',
        'select_race': 'Select Race to Review',
        'race_info': '📁 Race Info',
        'race_id': 'Race ID',
        'detection_issue': '⚠️ Detection Issue',
        'detected_duration': 'Detected Duration',
        'frames': 'Frames',
        'start_confidence': 'Start Confidence',
        'finish_confidence': 'Finish Confidence',
        'correction_status': '✅ Correction Status',
        'corrected': 'Corrected',
        'start_frame': 'Start Frame',
        'finish_frame': 'Finish Frame',
        'not_corrected': 'Not yet corrected',
        'correct_boundaries': '✏️ Correct Race Boundaries',
        'current_detected': 'Current detected',
        'mark_start': '📍 Mark Current as START',
        'mark_finish': '📍 Mark Current as FINISH',
        'marked_at': 'marked at frame {frame} ({time:.2f}s)',
        'new_start_frame': 'New Start Frame',
        'new_finish_frame': 'New Finish Frame',
        'new_duration_s': 'New Duration (s)',
        'new_duration_frames': 'New Duration (frames)',
        'valid_duration': '✅ Valid duration',
        'below_minimum': '⚠️ Below minimum ({min}s)',
        'above_maximum': '⚠️ Above maximum ({max}s)',
        'correction_details': '📝 Correction Details',
        'correction_reason': 'Correction Reason',
        'correction_reason_placeholder': 'e.g., Climber fall detected as finish, Pre-race warmup included, False start',
        'reviewer_notes': 'Reviewer Notes',
        'reviewer_notes_placeholder': 'Additional notes about this correction...',
        'save_correction': '💾 Save Correction',
        'skip_race': '⏭️ Skip Race',
        'reset': '🔄 Reset',
        'provide_reason': 'Please provide a correction reason!',
        'correction_saved': '✅ Correction saved successfully!',
        'race_skipped': 'Race skipped',
        'video_not_found': 'Video file not found',
        'competition_not_found': 'Competition config not found',
        'no_races': 'No races match the selected filters. Adjust filters in the sidebar.',
        'language_selector': 'Language / زبان',
        'settings': '⚙️ Settings',
        'about': 'ℹ️ About',
        'help': '❓ Help',
        'progress_percentage': 'Progress: {percent:.1f}%',
        'validation_results': 'Validation Results',
        'all_valid': '✅ All validations passed',
        'has_errors': '❌ Validation errors found',
        'critical_errors': 'Critical Errors',
        'warnings': 'Warnings',
        'page_selector': 'Page',
        'race_review_page': '🏁 Race Review',
        'video_library_page': '📚 Video Library',
        'video_library_title': '📚 Video Library - All Races',
        'library_subtitle': 'View and manage all race videos across all competitions',
        'filter_competition': 'Filter by Competition',
        'filter_status': 'Filter by Status',
        'search_videos': 'Search',
        'search_placeholder': 'Search race ID, athlete names, or notes...',
        'library_stats': 'Library Statistics',
        'reviewed': 'Reviewed',
        'suspicious': 'Suspicious',
        'failed': 'Failed',
        'video_table': 'Videos',
        'race_id_col': 'Race ID',
        'competition_col': 'Competition',
        'duration_col': 'Duration (s)',
        'status_col': 'Status',
        'athletes_col': 'Athletes',
        'notes_col': 'Notes',
        'no_videos_found': 'No videos found matching the filters',
        'quick_actions': 'Quick Actions',
        'select_race_action': 'Select race for action:',
        'view_in_player': '👁️ View in Player',
        'export_library': '📥 Export Library',
        'export_format': 'Export Format',
        'export_button': '📥 Export',
        'loading_library': 'Loading video library...',
        'add_video_page': '➕ Add Video',
        'bulk_ops_page': '⚡ Bulk Operations',
        'add_video_title': '➕ Add New Video - Manual Extraction',
        'add_video_subtitle': 'Extract race segments from source videos with manual timestamps',
        'bulk_ops_title': '⚡ Bulk Operations - Batch Processing',
        'bulk_ops_subtitle': 'Process multiple races with export, validation, and reporting',
    },
    'fa': {
        'page_title': '🏔️ سنگنوردی سرعتی - رابط بررسی دستی مسابقات',
        'subtitle': 'رفع خطاهای تشخیص مسابقه برای مسابقات مشکوک',
        'sidebar_stats': '📊 آمار پیشرفت',
        'total_races': 'مجموع مسابقات',
        'completed': 'تکمیل شده',
        'pending': 'در انتظار',
        'skipped': 'رد شده',
        'critical': 'بحرانی',
        'filter_races': '🎯 فیلتر مسابقات',
        'competition': 'مسابقه',
        'priority': 'اولویت',
        'status': 'وضعیت',
        'all': 'همه',
        'pending_only': 'فقط در انتظار',
        'race_review': '🏁 بررسی مسابقه',
        'showing_races': '📋 نمایش {count} مسابقه (از {total} مسابقه کل)',
        'select_race': 'انتخاب مسابقه برای بررسی',
        'race_info': '📁 اطلاعات مسابقه',
        'race_id': 'شناسه مسابقه',
        'detection_issue': '⚠️ مشکل تشخیص',
        'detected_duration': 'مدت زمان تشخیص داده شده',
        'frames': 'فریم‌ها',
        'start_confidence': 'اطمینان شروع',
        'finish_confidence': 'اطمینان پایان',
        'correction_status': '✅ وضعیت اصلاح',
        'corrected': 'اصلاح شده',
        'start_frame': 'فریم شروع',
        'finish_frame': 'فریم پایان',
        'not_corrected': 'هنوز اصلاح نشده',
        'correct_boundaries': '✏️ اصلاح محدوده مسابقه',
        'current_detected': 'تشخیص داده شده فعلی',
        'mark_start': '📍 علامت‌گذاری فریم فعلی به عنوان شروع',
        'mark_finish': '📍 علامت‌گذاری فریم فعلی به عنوان پایان',
        'marked_at': 'علامت‌گذاری شد در فریم {frame} ({time:.2f}s)',
        'new_start_frame': 'فریم شروع جدید',
        'new_finish_frame': 'فریم پایان جدید',
        'new_duration_s': 'مدت زمان جدید (ثانیه)',
        'new_duration_frames': 'مدت زمان جدید (فریم)',
        'valid_duration': '✅ مدت زمان معتبر',
        'below_minimum': '⚠️ کمتر از حداقل ({min}s)',
        'above_maximum': '⚠️ بیشتر از حداکثر ({max}s)',
        'correction_details': '📝 جزئیات اصلاح',
        'correction_reason': 'دلیل اصلاح',
        'correction_reason_placeholder': 'مثال: سقوط ورزشکار به عنوان پایان تشخیص داده شد، گرم کردن قبل از مسابقه شامل شد، استارت اشتباه',
        'reviewer_notes': 'یادداشت‌های بازبین',
        'reviewer_notes_placeholder': 'یادداشت‌های اضافی درباره این اصلاح...',
        'save_correction': '💾 ذخیره اصلاح',
        'skip_race': '⏭️ رد کردن مسابقه',
        'reset': '🔄 بازنشانی',
        'provide_reason': 'لطفاً دلیل اصلاح را وارد کنید!',
        'correction_saved': '✅ اصلاح با موفقیت ذخیره شد!',
        'race_skipped': 'مسابقه رد شد',
        'video_not_found': 'فایل ویدئو یافت نشد',
        'competition_not_found': 'پیکربندی مسابقه یافت نشد',
        'no_races': 'هیچ مسابقه‌ای با فیلترهای انتخابی مطابقت ندارد. فیلترها را در نوار کناری تنظیم کنید.',
        'language_selector': 'Language / زبان',
        'settings': '⚙️ تنظیمات',
        'about': 'ℹ️ درباره',
        'help': '❓ راهنما',
        'progress_percentage': 'پیشرفت: {percent:.1f}%',
        'validation_results': 'نتایج اعتبارسنجی',
        'all_valid': '✅ همه اعتبارسنجی‌ها موفق',
        'has_errors': '❌ خطاهای اعتبارسنجی یافت شد',
        'critical_errors': 'خطاهای بحرانی',
        'warnings': 'هشدارها',
        'page_selector': 'صفحه',
        'race_review_page': '🏁 بررسی مسابقات',
        'video_library_page': '📚 کتابخانه ویدئو',
        'video_library_title': '📚 کتابخانه ویدئو - همه مسابقات',
        'library_subtitle': 'مشاهده و مدیریت تمام ویدیوهای مسابقه در همه رقابت‌ها',
        'filter_competition': 'فیلتر بر اساس مسابقه',
        'filter_status': 'فیلتر بر اساس وضعیت',
        'search_videos': 'جستجو',
        'search_placeholder': 'جستجوی شناسه مسابقه، نام ورزشکاران یا یادداشت‌ها...',
        'library_stats': 'آمار کتابخانه',
        'reviewed': 'بررسی شده',
        'suspicious': 'مشکوک',
        'failed': 'ناموفق',
        'video_table': 'ویدیوها',
        'race_id_col': 'شناسه مسابقه',
        'competition_col': 'مسابقه',
        'duration_col': 'مدت زمان (ثانیه)',
        'status_col': 'وضعیت',
        'athletes_col': 'ورزشکاران',
        'notes_col': 'یادداشت‌ها',
        'no_videos_found': 'هیچ ویدیویی با فیلترها پیدا نشد',
        'quick_actions': 'اقدامات سریع',
        'select_race_action': 'انتخاب مسابقه برای اقدام:',
        'view_in_player': '👁️ مشاهده در پلیر',
        'export_library': '📥 خروجی کتابخانه',
        'export_format': 'فرمت خروجی',
        'export_button': '📥 دریافت',
        'loading_library': 'بارگذاری کتابخانه ویدئو...',
        'add_video_page': '➕ افزودن ویدئو',
        'bulk_ops_page': '⚡ عملیات دسته‌ای',
        'add_video_title': '➕ افزودن ویدئو جدید - استخراج دستی',
        'add_video_subtitle': 'استخراج بخش‌های مسابقه از ویدیوهای منبع با timestamps دستی',
        'bulk_ops_title': '⚡ عملیات دسته‌ای - پردازش گروهی',
        'bulk_ops_subtitle': 'پردازش چندین مسابقه با export، اعتبارسنجی و گزارش‌گیری',
    }
}


def get_text(key: str, **kwargs) -> str:
    """Get translated text based on current language."""
    lang = st.session_state.get('language', 'en')
    text = TRANSLATIONS[lang].get(key, key)
    return text.format(**kwargs) if kwargs else text


# =============================================================================
# PAGE CONFIGURATION / پیکربندی صفحه
# =============================================================================

st.set_page_config(
    page_title="Race Detection Review Interface",
    page_icon="🏔️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# INITIALIZE MANAGERS / مقداردهی اولیه مدیران
# =============================================================================

@st.cache_resource
def get_managers():
    """Initialize and cache managers."""
    config_mgr = ConfigManager()
    progress_tracker = ProgressTracker()
    metadata_mgr = MetadataManager()
    validator = RaceValidator()
    phase_mgr = PhaseManager(config_mgr)
    video_extractor = VideoExtractor(config_mgr)
    bulk_ops_mgr = BulkOperationsManager(config_mgr, metadata_mgr)
    return config_mgr, progress_tracker, metadata_mgr, validator, phase_mgr, video_extractor, bulk_ops_mgr


try:
    config_mgr, progress_tracker, metadata_mgr, validator, phase_mgr, video_extractor, bulk_ops_mgr = get_managers()
except Exception as e:
    st.error(f"Error initializing application: {e}")
    st.stop()


# =============================================================================
# LANGUAGE SELECTOR / انتخاب زبان
# =============================================================================

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

# Language selector in sidebar
with st.sidebar:
    st.markdown("---")
    selected_lang = st.selectbox(
        "Language / زبان",
        options=['en', 'fa'],
        format_func=lambda x: '🇬🇧 English' if x == 'en' else '🇮🇷 فارسی',
        key='language_selector'
    )
    if selected_lang != st.session_state['language']:
        st.session_state['language'] = selected_lang
        st.rerun()


# =============================================================================
# TITLE / عنوان
# =============================================================================

st.title(get_text('page_title'))
st.markdown(f"**{get_text('subtitle')}**")


# =============================================================================
# PAGE SELECTOR / انتخاب صفحه
# =============================================================================

# Initialize page in session state
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'race_review'

# Page selector
page_options = {
    'race_review': get_text('race_review_page'),
    'video_library': get_text('video_library_page'),
    'add_video': get_text('add_video_page'),
    'bulk_operations': get_text('bulk_ops_page')
}

selected_page = st.radio(
    get_text('page_selector'),
    options=list(page_options.keys()),
    format_func=lambda x: page_options[x],
    horizontal=True,
    key='page_selector_radio'
)

if selected_page != st.session_state['current_page']:
    st.session_state['current_page'] = selected_page
    st.rerun()

st.markdown("---")


# =============================================================================
# PAGE: RACE REVIEW / صفحه: بررسی مسابقات
# =============================================================================

if st.session_state['current_page'] == 'race_review':

    # SIDEBAR - Statistics & Filters
    with st.sidebar:
        st.header(get_text('sidebar_stats'))

        stats = progress_tracker.get_statistics()
        progress_pct = progress_tracker.get_progress_percentage()

        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('total_races'), stats['total'])
            st.metric(
                get_text('completed'),
                stats['completed'],
                delta=get_text('progress_percentage', percent=progress_pct)
            )
        with col2:
            st.metric(get_text('pending'), stats['pending'])
            st.metric(
                get_text('critical'),
                stats['critical'],
                delta="Priority 1",
                delta_color="inverse"
            )

        st.markdown("---")

        # Competition filter
        st.header(get_text('filter_races'))

        competitions = config_mgr.get_competitions()
        competition_names = [get_text('all')] + [c.name for c in competitions]
        selected_competition = st.selectbox(get_text('competition'), competition_names)

        # Priority filter
        priority_options = [
            get_text('all'),
            "Critical (1)",
            "High (2)",
            "Medium (3)",
            "Low (4)"
        ]
        selected_priority = st.selectbox(get_text('priority'), priority_options)

        # Status filter
        status_options = [
            get_text('pending_only'),
            get_text('all'),
            "Completed",
            "Skipped"
        ]
        selected_status = st.selectbox(get_text('status'), status_options)


    # =============================================================================
    # MAIN CONTENT - RACE LIST AND REVIEW / محتوای اصلی
    # =============================================================================

    st.header(get_text('race_review'))

    # Load races based on filters
    all_races = progress_tracker.load_all_races()

    # Apply filters
    filtered_races = all_races

    if selected_competition != get_text('all'):
        comp_key = next((c.key for c in competitions if c.name == selected_competition), None)
        if comp_key:
            filtered_races = [r for r in filtered_races if r.competition == comp_key]

    if selected_priority != get_text('all'):
        priority_num = int(selected_priority.split('(')[1].strip(')'))
        filtered_races = [r for r in filtered_races if r.priority == priority_num]

    if selected_status == get_text('pending_only'):
        filtered_races = [r for r in filtered_races if r.review_status == 'Pending']
    elif selected_status != get_text('all'):
        filtered_races = [r for r in filtered_races if r.review_status == selected_status]

    # Sort by priority
    filtered_races.sort(key=lambda r: (r.priority, r.race_id))

    st.info(get_text('showing_races', count=len(filtered_races), total=len(all_races)))


    # =============================================================================
    # RACE SELECTION AND REVIEW / انتخاب و بررسی مسابقه
    # =============================================================================

    if filtered_races:
        race_options = [
            f"[P{r.priority}] {r.race_id} ({r.detected_duration_s:.2f}s → {r.issue_description})"
            for r in filtered_races
        ]
        selected_race_idx = st.selectbox(
            get_text('select_race'),
            range(len(race_options)),
            format_func=lambda i: race_options[i]
        )

        selected_race = filtered_races[selected_race_idx]

        st.markdown("---")

        # Display race information
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(get_text('race_info'))
            st.text(f"{get_text('race_id')}: {selected_race.race_id}")
            st.text(f"{get_text('competition')}: {selected_race.competition}")
            st.text(f"{get_text('priority')}: {selected_race.priority}")
            st.text(f"{get_text('status')}: {selected_race.review_status}")

        with col2:
            st.subheader(get_text('detection_issue'))
            st.error(f"**{selected_race.issue_description}**")
            st.text(f"{get_text('detected_duration')}: {selected_race.detected_duration_s:.2f}s")
            st.text(f"{get_text('frames')}: {selected_race.duration_frames}")
            st.text(f"{get_text('start_confidence')}: {selected_race.confidence_start:.2f}")
            st.text(f"{get_text('finish_confidence')}: {selected_race.confidence_finish:.2f}")

        with col3:
            st.subheader(get_text('correction_status'))
            if selected_race.corrected_duration_s:
                st.success(f"{get_text('corrected')}: {selected_race.corrected_duration_s}s")
                st.text(f"{get_text('start_frame')}: {selected_race.corrected_start_frame}")
                st.text(f"{get_text('finish_frame')}: {selected_race.corrected_finish_frame}")
            else:
                st.warning(get_text('not_corrected'))

        st.markdown("---")

        # Load video and metadata
        competition_config = config_mgr.get_competition(selected_race.competition)

        if competition_config:
            video_path = metadata_mgr.get_video_path(selected_race.competition, selected_race.race_id)

            if video_path.exists():
                # Load current metadata
                try:
                    metadata = metadata_mgr.load_metadata(selected_race.competition, selected_race.race_id)

                    # Video player
                    player = VideoPlayer(video_path, competition_config.fps)
                    current_frame, current_time = player.render(
                        key_prefix=f"race_{selected_race.race_id}",
                        language=st.session_state['language']
                    )

                    st.markdown("---")

                    # Correction interface
                    st.subheader(get_text('correct_boundaries'))

                    # Warning if detected frames are from original video
                    if (metadata.get('detected_start_frame', 0) >= player.total_frames or
                        metadata.get('detected_finish_frame', 0) >= player.total_frames):
                        st.warning(
                            "⚠️ **Note**: Detected frames are from the original video (not this extracted segment). "
                            "Please use the video player to find the correct start and finish frames in this segment."
                        )

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown(f"**🟢 {get_text('start_frame')}**")
                        st.text(f"{get_text('current_detected')}: {metadata['detected_start_frame']}")

                        if st.button(get_text('mark_start'), key="mark_start"):
                            st.session_state['new_start_frame'] = current_frame
                            st.success(get_text('marked_at', frame=current_frame, time=current_time))

                        # Clamp detected frame to valid range for extracted video
                        default_start = metadata.get('detected_start_frame', 0)
                        if default_start >= player.total_frames:
                            default_start = 0  # Reset to beginning if out of range

                        new_start_frame = st.number_input(
                            get_text('new_start_frame'),
                            min_value=0,
                            max_value=player.total_frames - 1,
                            value=st.session_state.get('new_start_frame', default_start),
                            key="start_frame_input"
                        )

                    with col2:
                        st.markdown(f"**🔴 {get_text('finish_frame')}**")
                        st.text(f"{get_text('current_detected')}: {metadata['detected_finish_frame']}")

                        if st.button(get_text('mark_finish'), key="mark_finish"):
                            st.session_state['new_finish_frame'] = current_frame
                            st.success(get_text('marked_at', frame=current_frame, time=current_time))

                        # Clamp detected frame to valid range for extracted video
                        default_finish = metadata.get('detected_finish_frame', player.total_frames - 1)
                        if default_finish >= player.total_frames:
                            default_finish = player.total_frames - 1  # Reset to end if out of range

                        new_finish_frame = st.number_input(
                            get_text('new_finish_frame'),
                            min_value=0,
                            max_value=player.total_frames - 1,
                            value=st.session_state.get('new_finish_frame', default_finish),
                            key="finish_frame_input"
                        )

                    # Calculate new duration
                    new_duration_frames = new_finish_frame - new_start_frame
                    new_duration_seconds = new_duration_frames / competition_config.fps

                    st.markdown("---")

                    # Validation
                    validation_results = validator.validate_all(
                        new_start_frame,
                        new_finish_frame,
                        competition_config.fps,
                        player.total_frames
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(get_text('new_duration_s'), f"{new_duration_seconds:.2f}")
                    with col2:
                        st.metric(get_text('new_duration_frames'), new_duration_frames)
                    with col3:
                        # Check validation
                        validation_rules = config_mgr.get_validation_rules()
                        min_dur = validation_rules['duration']['min']
                        max_dur = validation_rules['duration']['max']

                        if new_duration_seconds < min_dur:
                            st.error(get_text('below_minimum', min=min_dur))
                        elif new_duration_seconds > max_dur:
                            st.error(get_text('above_maximum', max=max_dur))
                        else:
                            st.success(get_text('valid_duration'))

                    # Display validation results
                    st.subheader(get_text('validation_results'))
                    if validator.is_all_valid(validation_results):
                        st.success(get_text('all_valid'))
                    else:
                        st.warning(get_text('has_errors'))

                        # Show critical errors
                        critical_errors = validator.get_critical_errors(validation_results)
                        if critical_errors:
                            st.error(f"**{get_text('critical_errors')}:**")
                            for err in critical_errors:
                                st.error(f"- {err}")

                        # Show warnings
                        warnings = validator.get_warnings(validation_results)
                        if warnings:
                            st.warning(f"**{get_text('warnings')}:**")
                            for warn in warnings:
                                st.warning(f"- {warn}")

                    # Correction reason
                    st.markdown("---")
                    st.subheader(get_text('correction_details'))

                    correction_reason = st.text_input(
                        get_text('correction_reason'),
                        value="",
                        placeholder=get_text('correction_reason_placeholder')
                    )

                    reviewer_notes = st.text_area(
                        get_text('reviewer_notes'),
                        value="",
                        placeholder=get_text('reviewer_notes_placeholder')
                    )

                    # Save buttons
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button(get_text('save_correction'), type="primary", key="save_correction"):
                            if not correction_reason:
                                st.error(get_text('provide_reason'))
                            else:
                                # Update metadata
                                updated_metadata = metadata_mgr.update_race_boundaries(
                                    competition=selected_race.competition,
                                    race_id=selected_race.race_id,
                                    new_start_frame=int(new_start_frame),
                                    new_finish_frame=int(new_finish_frame),
                                    fps=competition_config.fps,
                                    correction_reason=correction_reason,
                                    reviewer_notes=reviewer_notes
                                )

                                # Update progress tracker
                                selected_race.review_status = 'Completed'
                                selected_race.corrected_duration_s = str(new_duration_seconds)
                                selected_race.corrected_start_frame = str(int(new_start_frame))
                                selected_race.corrected_finish_frame = str(int(new_finish_frame))
                                selected_race.reviewer_notes = reviewer_notes
                                selected_race.review_date = updated_metadata['correction_metadata']['correction_date']

                                progress_tracker.update_race(selected_race)

                                st.success(get_text('correction_saved'))
                                st.balloons()

                                # Clear session state
                                if 'new_start_frame' in st.session_state:
                                    del st.session_state['new_start_frame']
                                if 'new_finish_frame' in st.session_state:
                                    del st.session_state['new_finish_frame']

                                st.rerun()

                    with col2:
                        if st.button(get_text('skip_race'), key="skip_race"):
                            selected_race.review_status = 'Skipped'
                            progress_tracker.update_race(selected_race)
                            st.info(get_text('race_skipped'))
                            st.rerun()

                    with col3:
                        if st.button(get_text('reset'), key="reset_form"):
                            if 'new_start_frame' in st.session_state:
                                del st.session_state['new_start_frame']
                            if 'new_finish_frame' in st.session_state:
                                del st.session_state['new_finish_frame']
                            st.rerun()

                    # Cleanup
                    player.close()

                except Exception as e:
                    st.error(f"Error loading race data: {e}")

            else:
                st.error(f"{get_text('video_not_found')}: {video_path}")
        else:
            st.error(f"{get_text('competition_not_found')}: {selected_race.competition}")

    else:
        st.info(get_text('no_races'))


# =============================================================================
# PAGE: VIDEO LIBRARY / صفحه: کتابخانه ویدئو
# =============================================================================

elif st.session_state['current_page'] == 'video_library':

    st.header(get_text('video_library_title'))
    st.markdown(f"**{get_text('library_subtitle')}**")

    # Initialize VideoLibrary
    video_lib = VideoLibrary(config_mgr)

    with st.spinner(get_text('loading_library')):
        all_videos = video_lib.get_all_videos()

    st.markdown("---")

    # Filters in sidebar
    with st.sidebar:
        st.header(get_text('library_stats'))

        # Get statistics
        stats = video_lib.get_statistics(all_videos)

        col1, col2 = st.columns(2)
        with col1:
            st.metric(get_text('total_races'), stats['total'])
            st.metric(get_text('reviewed'), stats['by_status'].get('reviewed', 0))
        with col2:
            st.metric(get_text('suspicious'), stats['by_status'].get('suspicious', 0))
            st.metric(get_text('pending'), stats['by_status'].get('pending', 0))

        st.markdown("---")

        # Filters
        st.subheader(get_text('filter_competition'))
        competitions = config_mgr.get_competitions()
        comp_options = ['all'] + [c.key for c in competitions]
        selected_comp_filter = st.selectbox(
            get_text('competition'),
            comp_options,
            format_func=lambda x: get_text('all') if x == 'all' else x
        )

        st.subheader(get_text('filter_status'))
        status_options = ['all', 'reviewed', 'suspicious', 'pending', 'failed']
        selected_status_filter = st.selectbox(
            get_text('status'),
            status_options,
            format_func=lambda x: get_text(x) if x != 'all' else get_text('all')
        )

        st.subheader(get_text('search_videos'))
        search_query = st.text_input(
            get_text('search_videos'),
            placeholder=get_text('search_placeholder'),
            label_visibility='collapsed'
        )

    # Apply filters
    filtered_videos = video_lib.filter_videos(
        all_videos,
        competition=selected_comp_filter,
        status=selected_status_filter if selected_status_filter != 'all' else None,
        search_query=search_query if search_query else None
    )

    # Display filtered count
    st.info(f"Showing {len(filtered_videos)} videos (filtered from {len(all_videos)} total)")

    # Display videos in table
    if filtered_videos:
        import pandas as pd

        # Create dataframe
        df_data = []
        for video in filtered_videos:
            # Handle None values
            notes_text = video.notes if video.notes else ""
            notes_display = notes_text[:40] + '...' if len(notes_text) > 40 else notes_text

            df_data.append({
                get_text('race_id_col'): video.race_id,
                get_text('competition_col'): video.competition,
                get_text('duration_col'): f"{video.duration:.2f}",
                get_text('status_col'): get_text(video.status) if video.status in ['reviewed', 'suspicious', 'pending', 'failed'] else video.status,
                get_text('athletes_col'): f"{video.left_athlete} vs {video.right_athlete}",
                get_text('notes_col'): notes_display
            })

        df = pd.DataFrame(df_data)

        # Display table
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            height=400
        )

        # Quick actions
        st.markdown("---")
        st.subheader(get_text('quick_actions'))

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_race_id = st.selectbox(
                get_text('select_race_action'),
                [v.race_id for v in filtered_videos]
            )
        with col2:
            if st.button(get_text('view_in_player')):
                # Switch to race review page with this race selected
                st.session_state['current_page'] = 'race_review'
                st.session_state['selected_race_id'] = selected_race_id
                st.rerun()

        # Export functionality
        st.markdown("---")
        st.subheader(get_text('export_library'))

        col1, col2 = st.columns([2, 1])
        with col1:
            export_format = st.selectbox(
                get_text('export_format'),
                ['json', 'csv', 'yaml']
            )
        with col2:
            if st.button(get_text('export_button')):
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = Path(f"video_library_export_{timestamp}.{export_format}")

                success, message = video_lib.export_library_info(
                    filtered_videos,
                    output_path,
                    export_format
                )

                if success:
                    st.success(message)
                    # Provide download button
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label=f"Download {output_path.name}",
                            data=f.read(),
                            file_name=output_path.name,
                            mime=f"application/{export_format}"
                        )
                else:
                    st.error(message)

    else:
        st.warning(get_text('no_videos_found'))


# =============================================================================
# PAGE: ADD VIDEO / صفحه: افزودن ویدئو
# =============================================================================

elif st.session_state['current_page'] == 'add_video':

    st.header(get_text('add_video_title'))
    st.markdown(f"**{get_text('add_video_subtitle')}**")
    st.markdown("---")

    # Check FFmpeg availability
    ffmpeg_available, ffmpeg_msg = video_extractor.check_ffmpeg_available()

    if not ffmpeg_available:
        st.error(f"❌ {ffmpeg_msg}")
        st.info("Install FFmpeg to use this feature:")
        st.code("# Ubuntu/Debian\nsudo apt-get install ffmpeg\n\n# macOS\nbrew install ffmpeg\n\n# Windows\nDownload from https://ffmpeg.org/download.html")
        st.stop()
    else:
        st.success(f"✅ {ffmpeg_msg}")

    st.markdown("---")

    # Step 1: Select competition
    st.subheader("1️⃣ Competition Selection")
    competitions = config_mgr.get_competitions()
    comp_options = {c.key: f"{c.name} ({c.date})" for c in competitions}

    selected_comp_key = st.selectbox(
        "Select Competition",
        options=list(comp_options.keys()),
        format_func=lambda x: comp_options[x]
    )

    comp_config = config_mgr.get_competition(selected_comp_key)

    # Step 2: Source video selection
    st.subheader("2️⃣ Source Video")

    col1, col2 = st.columns([3, 1])
    with col1:
        source_video_path = st.text_input(
            "Source Video Path",
            value="",
            placeholder="/path/to/source_video.mp4"
        )
    with col2:
        # Suggest race ID
        suggested_race_id = video_extractor.suggest_race_id(selected_comp_key)
        st.text_input("Suggested Race ID", value=suggested_race_id, disabled=True)

    # Get video info if path provided
    if source_video_path and Path(source_video_path).exists():
        video_info = video_extractor.get_video_info(Path(source_video_path))
        if video_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", f"{video_info.get('duration', 0):.1f}s")
            with col2:
                st.metric("FPS", f"{video_info.get('fps', 0):.1f}")
            with col3:
                st.metric("Resolution", video_info.get('resolution', 'Unknown'))

    # Step 3: Race information
    st.subheader("3️⃣ Race Information")

    col1, col2 = st.columns(2)
    with col1:
        race_id = st.text_input(
            "Race ID",
            value=suggested_race_id,
            help="Unique identifier for this race"
        )
    with col2:
        round_name = st.text_input(
            "Round Name",
            value="",
            placeholder="e.g., 1/8 final - Men"
        )

    # Step 4: Timestamps
    st.subheader("4️⃣ Manual Timestamps")
    st.caption("Enter start and finish times in MM:SS or HH:MM:SS format")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_time = st.text_input(
            "Start Time",
            value="",
            placeholder="01:30 or 00:01:30",
            help="Time when race starts (MM:SS or HH:MM:SS)"
        )
    with col2:
        end_time = st.text_input(
            "End Time",
            value="",
            placeholder="01:36 or 00:01:36",
            help="Time when race finishes"
        )
    with col3:
        buffer_before = st.number_input(
            "Buffer Before (s)",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Seconds to include before start"
        )
    with col4:
        buffer_after = st.number_input(
            "Buffer After (s)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.5,
            help="Seconds to include after finish"
        )

    # Validate timestamps
    if start_time and end_time:
        valid_start, msg_start, start_seconds = video_extractor.validate_timestamp(start_time)
        valid_end, msg_end, end_seconds = video_extractor.validate_timestamp(end_time)

        if not valid_start:
            st.error(f"❌ Start time error: {msg_start}")
        elif not valid_end:
            st.error(f"❌ End time error: {msg_end}")
        elif end_seconds <= start_seconds:
            st.error("❌ End time must be after start time")
        else:
            race_duration = end_seconds - start_seconds
            st.success(f"✅ Valid timestamps - Race duration: {race_duration:.2f}s")

    # Step 5: Athlete information
    st.subheader("5️⃣ Athlete Information")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Left Lane**")
        left_name = st.text_input("Athlete Name (Left)", value="", placeholder="John Doe")
        left_country = st.text_input("Country (Left)", value="", placeholder="USA")
        left_bib = st.selectbox("Bib Color (Left)", ["red", "blue", "yellow", "green"], index=0)

    with col2:
        st.markdown("**Right Lane**")
        right_name = st.text_input("Athlete Name (Right)", value="", placeholder="Jane Smith")
        right_country = st.text_input("Country (Right)", value="", placeholder="FRA")
        right_bib = st.selectbox("Bib Color (Right)", ["blue", "red", "yellow", "green"], index=1)

    # Step 6: Extract button
    st.markdown("---")
    st.subheader("6️⃣ Extract Race Segment")

    # Validation before extraction
    can_extract = (
        source_video_path and Path(source_video_path).exists() and
        race_id and start_time and end_time and
        left_name and right_name
    )

    if not can_extract:
        st.warning("⚠️ Please fill in all required fields: Source Video, Race ID, Timestamps, and Athlete Names")

    if st.button("🎬 Extract Race Segment", type="primary", disabled=not can_extract):
        with st.spinner("Extracting race segment with FFmpeg..."):
            # Prepare athlete dicts
            left_athlete = {
                "name": left_name,
                "country": left_country,
                "bib_color": left_bib
            }
            right_athlete = {
                "name": right_name,
                "country": right_country,
                "bib_color": right_bib
            }

            # Extract segment
            success, message, output_path = video_extractor.extract_manual_segment(
                source_video=Path(source_video_path),
                competition_key=selected_comp_key,
                race_id=race_id,
                start_time=start_time,
                end_time=end_time,
                left_athlete=left_athlete,
                right_athlete=right_athlete,
                round_name=round_name,
                buffer_before=buffer_before,
                buffer_after=buffer_after
            )

            if success:
                st.success(f"✅ {message}")
                st.balloons()
                st.info(f"📁 Video saved to: {output_path}")
                st.info(f"📝 Metadata saved to: {output_path.with_name(f'{race_id}_metadata.json')}")

                # Clear inputs
                if st.button("🔄 Extract Another Race"):
                    st.rerun()
            else:
                st.error(f"❌ {message}")


# =============================================================================
# PAGE: BULK OPERATIONS / صفحه: عملیات دسته‌ای
# =============================================================================

elif st.session_state['current_page'] == 'bulk_operations':

    st.header(get_text('bulk_ops_title'))
    st.markdown(f"**{get_text('bulk_ops_subtitle')}**")
    st.markdown("---")

    # Initialize VideoLibrary for race selection
    video_lib = VideoLibrary(config_mgr)
    all_videos = video_lib.get_all_videos()

    # Step 1: Race Selection
    st.subheader("1️⃣ Select Races")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Filter by competition
        comp_options = ['all'] + [c.key for c in config_mgr.get_competitions()]
        selected_comp = st.selectbox(
            "Filter by Competition",
            comp_options,
            format_func=lambda x: "All Competitions" if x == 'all' else x
        )

    with col2:
        # Filter by status
        status_options = ['all', 'reviewed', 'suspicious', 'pending', 'failed']
        selected_status = st.selectbox(
            "Filter by Status",
            status_options,
            format_func=lambda x: "All Statuses" if x == 'all' else x.title()
        )

    with col3:
        # Search
        search_query = st.text_input(
            "Search",
            placeholder="Race ID, athlete names..."
        )

    # Apply filters
    filtered_videos = video_lib.filter_videos(
        all_videos,
        competition=selected_comp if selected_comp != 'all' else None,
        status=selected_status if selected_status != 'all' else None,
        search_query=search_query if search_query else None
    )

    st.info(f"Selected {len(filtered_videos)} races (from {len(all_videos)} total)")

    # Display selected races
    if filtered_videos:
        with st.expander(f"📋 View Selected Races ({len(filtered_videos)})", expanded=False):
            race_list = "\n".join([f"- {v.race_id} ({v.competition})" for v in filtered_videos[:20]])
            if len(filtered_videos) > 20:
                race_list += f"\n... and {len(filtered_videos) - 20} more"
            st.text(race_list)

    st.markdown("---")

    # Step 2: Operation Type
    st.subheader("2️⃣ Select Operation")

    operation_type = st.radio(
        "Operation Type",
        ["Export", "Validate", "Summary Report"],
        horizontal=True
    )

    st.markdown("---")

    # Step 3: Operation-specific options
    st.subheader("3️⃣ Configure Operation")

    if operation_type == "Export":
        col1, col2 = st.columns(2)

        with col1:
            export_format = st.selectbox(
                "Export Format",
                ['json', 'csv', 'yaml', 'npz'],
                format_func=lambda x: {
                    'json': 'JSON (Full data)',
                    'csv': 'CSV (Spreadsheet)',
                    'yaml': 'YAML (Config-friendly)',
                    'npz': 'NPZ (ML-ready NumPy)'
                }[x]
            )

        with col2:
            include_metrics = st.checkbox(
                "Include Performance Metrics",
                value=False,
                help="Include calculated metrics if available (velocity, acceleration, etc.)"
            )

        st.caption(f"**Format info**: {{\n'json': 'Full metadata for each race',\n'csv': 'Tabular format for Excel/Sheets',\n'yaml': 'Human-readable config format',\n'npz': 'NumPy arrays for ML training'\n}}['{export_format}']")

    elif operation_type == "Validate":
        st.info("Batch validation will check all selected races for issues")
        validation_strict = st.checkbox("Strict Validation", value=False, help="Enable stricter validation rules")

    elif operation_type == "Summary Report":
        st.info("Generate statistical summary of selected races")
        include_plots = st.checkbox("Include Visualizations", value=False, help="Add charts (requires matplotlib)")

    st.markdown("---")

    # Step 4: Execute
    st.subheader("4️⃣ Execute Operation")

    if not filtered_videos:
        st.warning("⚠️ No races selected. Adjust filters to select races.")
    else:
        if st.button(f"▶️ Execute {operation_type}", type="primary"):
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            with st.spinner(f"Processing {len(filtered_videos)} races..."):

                if operation_type == "Export":
                    output_filename = f"bulk_export_{selected_comp}_{timestamp}.{export_format}"
                    output_path = Path(output_filename)

                    success, message = bulk_ops_mgr.export_multiple_races(
                        filtered_videos,
                        export_format,
                        output_path,
                        include_metrics=include_metrics
                    )

                    if success:
                        st.success(f"✅ {message}")

                        # Provide download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label=f"📥 Download {output_filename}",
                                data=f.read(),
                                file_name=output_filename,
                                mime=f"application/{export_format}"
                            )
                    else:
                        st.error(f"❌ {message}")

                elif operation_type == "Validate":
                    results = bulk_ops_mgr.validate_multiple_races(filtered_videos, validator)

                    # Display results
                    success_count = sum(1 for r in results if r.success)
                    failed_count = len(results) - success_count

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("✅ Valid", success_count)
                    with col2:
                        st.metric("❌ Invalid", failed_count)

                    # Show details
                    with st.expander("📊 Validation Details"):
                        for result in results:
                            if result.success:
                                st.success(f"✅ {result.race_id}: {result.message}")
                            else:
                                st.error(f"❌ {result.race_id}: {result.message}")
                                if result.data and 'issues' in result.data:
                                    for issue in result.data['issues']:
                                        st.caption(f"  - {issue}")

                elif operation_type == "Summary Report":
                    output_filename = f"summary_report_{selected_comp}_{timestamp}.json"
                    output_path = Path(output_filename)

                    success, message = bulk_ops_mgr.generate_summary_report(
                        filtered_videos,
                        output_path
                    )

                    if success:
                        st.success(f"✅ {message}")

                        # Display summary
                        import json
                        with open(output_path, 'r') as f:
                            report = json.load(f)

                        st.subheader("📊 Summary Statistics")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Races", report['total_races'])
                        with col2:
                            st.metric("Competitions", len(report['summary']['by_competition']))
                        with col3:
                            if report['summary']['duration_stats']:
                                st.metric("Avg Duration", f"{report['summary']['duration_stats']['mean']:.2f}s")

                        # Show breakdown
                        with st.expander("🔍 Detailed Breakdown"):
                            st.json(report['summary'])

                        # Download button
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label=f"📥 Download Full Report",
                                data=f.read(),
                                file_name=output_filename,
                                mime="application/json"
                            )
                    else:
                        st.error(f"❌ {message}")


# =============================================================================
# FOOTER / پاورقی
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.8em;'>
    Speed Climbing Performance Analysis - Manual Review Interface v1.0<br>
    تحلیل عملکرد سنگنوردی سرعتی - رابط بررسی دستی نسخه 1.0
    </div>
    """,
    unsafe_allow_html=True
)
