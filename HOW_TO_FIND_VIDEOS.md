# How to Find and Download IFSC Speed Climbing Videos

## Quick Guide

Since we cannot automatically download videos, please follow these steps to get test videos:

---

## Option 1: IFSC Official YouTube Channel (Recommended)

### Step 1: Find Videos

1. **Go to IFSC Official Channel**: https://www.youtube.com/@sportclimbing

2. **Search for recent finals:**
   - In the channel's search bar, type: `"speed final 2024"`
   - Or browse: Videos → Filter by "Most Popular" or "Date"

3. **Look for these types of videos:**
   - "Speed Final - IFSC World Cup [City Name] 2024"
   - "Men's/Women's Speed - Paris 2024 Olympics"
   - "Speed Climbing World Championships 2023/2024"

4. **Identify dual-lane races:**
   - Thumbnail should show TWO climbers side by side
   - Title usually contains: "Final", "Semi-Final", or "Race"
   - Duration: typically 2-5 minutes per race

### Step 2: Download Videos

#### Method A: Using our downloader script

```bash
# Download a single video
python -m src.utils.youtube_downloader "https://youtube.com/watch?v=VIDEO_ID" 720p "my_race_name"

# Example:
python -m src.utils.youtube_downloader "https://youtube.com/watch?v=abc123" 720p "ifsc_final_2024_men"
```

#### Method B: Update the config file

1. Edit `configs/youtube_urls.yaml`
2. Replace `YOUR_VIDEO_ID_HERE` with actual video IDs
3. Run: `python scripts/download_priority_videos.py`

---

## Option 2: Use Sample/Test Videos

If you already have speed climbing videos on your computer:

1. **Place videos in**: `data/raw_videos/`

2. **Supported formats**: MP4, AVI, MOV

3. **Recommended specs:**
   - Resolution: 720p or higher
   - FPS: 30+ (60 FPS preferred)
   - Duration: Any (we'll detect race start/finish automatically)
   - Content: Dual-lane races (two climbers visible)

---

## Option 3: Record from Live Streams

IFSC streams competitions live on YouTube:

1. **Check schedule**: https://www.ifsc-climbing.org/calendar
2. **Live stream**: https://www.youtube.com/@sportclimbing/streams
3. **Use screen recorder** to capture:
   - OBS Studio (free): https://obsproject.com/
   - Nvidia ShadowPlay (if you have Nvidia GPU)
   - Windows Game Bar (Win+G)

---

## Recommended Test Videos

Based on 2024 season (you'll need to find exact URLs):

### Priority 1: Paris 2024 Olympics
- **Event**: Speed Climbing Finals
- **Date**: August 2024
- **Why**: Highest quality footage, Olympic standards
- **Search**: "Paris 2024 Speed Climbing Final" on IFSC channel

### Priority 2: World Cup Seoul 2024 (Season Finale)
- **Event**: IFSC World Cup Final
- **Date**: October 4-8, 2024
- **Why**: Season-ending competition with top athletes
- **Search**: "IFSC World Cup Seoul 2024 Speed Final"

### Priority 3: World Cup Chamonix 2024
- **Event**: IFSC World Cup
- **Date**: July 2024
- **Why**: Good camera angles, clear wall view
- **Search**: "IFSC World Cup Chamonix 2024 Speed"

---

## Video Quality Requirements

For best results, your videos should have:

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| Resolution | 720p (1280x720) | 1080p (1920x1080) |
| FPS | 30 | 60 |
| Bitrate | 2 Mbps | 5+ Mbps |
| Audio | Optional | Yes (for start beep detection) |
| Lighting | Good | Excellent |
| Camera Angle | Front view | Fixed front angle |

---

## Testing Your Videos

After downloading, test if they work:

```bash
# Check video properties
python -c "import cv2; cap = cv2.VideoCapture('data/raw_videos/your_video.mp4'); print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}'); print(f'Size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}')"

# Quick preview
python -c "import cv2; cap = cv2.VideoCapture('data/raw_videos/your_video.mp4'); ret, frame = cap.read(); cv2.imshow('Test', frame); cv2.waitKey(0)"
```

---

## Current Status

**Videos Available**:
- ❌ No videos downloaded yet

**Next Steps**:
1. Find 1-2 dual-lane race videos using the methods above
2. Download to `data/raw_videos/`
3. Run the dual-lane detection system
4. Review analysis results

---

## Troubleshooting

### "Video unavailable" error when downloading
- Video might be private, deleted, or region-locked
- Try a different video from the IFSC channel
- Check if you can watch it in your browser first

### "No JavaScript runtime" warning
- This is a yt-dlp warning, usually not critical
- Videos should still download successfully
- If downloads fail, install Node.js: https://nodejs.org/

### Download is very slow
- Change quality to 480p: `quality="480p"`
- Check your internet connection
- IFSC videos are large (100-500 MB)

---

## Copyright Notice

**IMPORTANT**:
- Videos are property of IFSC and respective creators
- Use only for personal research/education
- Do not redistribute or publish
- Always credit original source

---

## Need Help?

- IFSC Official Website: https://www.ifsc-climbing.org/
- IFSC YouTube: https://www.youtube.com/@sportclimbing
- Project Issues: Create an issue in this repository

---

**Last Updated**: 2025-11-12
