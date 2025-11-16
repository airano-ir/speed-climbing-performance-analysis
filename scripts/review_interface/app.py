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
    return config_mgr, progress_tracker, metadata_mgr, validator


try:
    config_mgr, progress_tracker, metadata_mgr, validator = get_managers()
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
# SIDEBAR - STATISTICS & NAVIGATION / نوار کناری - آمار و ناوبری
# =============================================================================

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

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**🟢 {get_text('start_frame')}**")
                    st.text(f"{get_text('current_detected')}: {metadata['detected_start_frame']}")

                    if st.button(get_text('mark_start'), key="mark_start"):
                        st.session_state['new_start_frame'] = current_frame
                        st.success(get_text('marked_at', frame=current_frame, time=current_time))

                    new_start_frame = st.number_input(
                        get_text('new_start_frame'),
                        min_value=0,
                        max_value=player.total_frames - 1,
                        value=st.session_state.get('new_start_frame', metadata['detected_start_frame']),
                        key="start_frame_input"
                    )

                with col2:
                    st.markdown(f"**🔴 {get_text('finish_frame')}**")
                    st.text(f"{get_text('current_detected')}: {metadata['detected_finish_frame']}")

                    if st.button(get_text('mark_finish'), key="mark_finish"):
                        st.session_state['new_finish_frame'] = current_frame
                        st.success(get_text('marked_at', frame=current_frame, time=current_time))

                    new_finish_frame = st.number_input(
                        get_text('new_finish_frame'),
                        min_value=0,
                        max_value=player.total_frames - 1,
                        value=st.session_state.get('new_finish_frame', metadata['detected_finish_frame']),
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
