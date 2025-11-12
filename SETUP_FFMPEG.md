# 🎬 FFmpeg Setup Guide for Windows

FFmpeg is required for audio/video processing in this project.

## ⚠️ Important
FFmpeg is **NOT currently installed** on your system. You have two options:

---

## Option 1: Quick Install with Chocolatey (Recommended)

If you have Chocolatey package manager:

```powershell
# Open PowerShell as Administrator
choco install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

---

## Option 2: Manual Installation

### Step 1: Download FFmpeg

1. Go to: https://www.gyan.dev/ffmpeg/builds/
2. Download: **ffmpeg-release-essentials.zip** (latest version)
3. Extract to: `C:\ffmpeg`

### Step 2: Add to PATH

1. Open **System Properties** → **Environment Variables**
2. Under **System Variables**, find `Path`
3. Click **Edit** → **New**
4. Add: `C:\ffmpeg\bin`
5. Click **OK** on all dialogs

### Step 3: Verify Installation

Open a **NEW** terminal and run:
```bash
ffmpeg -version
```

You should see FFmpeg version info.

---

## Option 3: Use Without FFmpeg (Limited Functionality)

If you cannot install FFmpeg, the project will still work but with limitations:
- ❌ No audio analysis (beep detection disabled)
- ❌ No audio extraction from videos
- ✅ Video processing still works
- ✅ Pose estimation still works

To proceed without FFmpeg:
1. Set `USE_AUDIO = False` in config
2. Skip audio-related features

---

## Testing FFmpeg

After installation, test with:

```bash
# Check version
ffmpeg -version

# Test audio extraction (if you have a video)
ffmpeg -i test_video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 test_audio.wav
```

---

## Next Steps

Once FFmpeg is installed:

1. Close and reopen your terminal
2. Run: `ffmpeg -version` to verify
3. Continue with: `pip install -r requirements_phase1_extended.txt`

---

## Troubleshooting

### "ffmpeg not found" after installation
- Make sure you opened a **NEW** terminal after adding to PATH
- Verify PATH with: `echo %PATH%` (should contain `C:\ffmpeg\bin`)
- Try logging out and back in

### Permission errors
- Run PowerShell/CMD as Administrator
- Check that `C:\ffmpeg` is not read-only

### Still not working?
- Use Option 3 (without FFmpeg) for now
- We can use cloud processing for audio analysis later

---

**Note:** This guide was auto-generated. FFmpeg status: ❌ NOT INSTALLED
