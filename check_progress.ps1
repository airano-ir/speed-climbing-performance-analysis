# Check Race Segmentation Progress
# Run this script to see the status of all competitions

Write-Host "=== Race Segments Status ===" -ForegroundColor Cyan
Write-Host ""

$competitions = @{
    'seoul_2024' = 31
    'villars_2024' = 24
    'chamonix_2024' = 32
    'innsbruck_2024' = 32
    'zilina_2025' = 69
}

$totalProcessed = 0
$totalExpected = 0

foreach ($comp in $competitions.Keys | Sort-Object) {
    $expected = $competitions[$comp]
    $totalExpected += $expected

    $path = "data\race_segments\$comp"

    if (Test-Path $path) {
        $mp4Files = Get-ChildItem -Path $path -Filter "*.mp4" -File
        $count = ($mp4Files | Measure-Object).Count
        $totalProcessed += $count

        # Get creation time of newest file to check if recently processed
        $newestFile = $mp4Files | Sort-Object CreationTime -Descending | Select-Object -First 1
        $age = if ($newestFile) {
            $timeSpan = (Get-Date) - $newestFile.CreationTime
            if ($timeSpan.TotalMinutes -lt 60) {
                "{0:N0} minutes ago" -f $timeSpan.TotalMinutes
            } elseif ($timeSpan.TotalHours -lt 24) {
                "{0:N1} hours ago" -f $timeSpan.TotalHours
            } else {
                "{0:N0} days ago" -f $timeSpan.TotalDays
            }
        } else {
            "N/A"
        }

        $status = if ($count -eq $expected) {
            "COMPLETE"
        } elseif ($count -gt 0) {
            "IN PROGRESS ($count/$expected)"
        } else {
            "NOT STARTED"
        }

        $color = if ($count -eq $expected) {
            'Green'
        } elseif ($count -gt 0) {
            'Yellow'
        } else {
            'Red'
        }

        Write-Host "$comp`: " -NoNewline
        Write-Host "$count/$expected races" -NoNewline -ForegroundColor White
        Write-Host " - " -NoNewline
        Write-Host $status -ForegroundColor $color -NoNewline
        if ($newestFile) {
            Write-Host " (newest: $age)" -ForegroundColor Gray
        } else {
            Write-Host ""
        }
    } else {
        Write-Host "$comp`: " -NoNewline
        Write-Host "NOT STARTED (directory doesn't exist)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Total Progress: $totalProcessed/$totalExpected races" -ForegroundColor Cyan
$percentage = [math]::Round(($totalProcessed / $totalExpected) * 100, 1)
Write-Host "Completion: $percentage%" -ForegroundColor Cyan

# Check if batch process is running
Write-Host ""
Write-Host "=== Process Status ===" -ForegroundColor Cyan
$pythonProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*python*" }
if ($pythonProcs) {
    Write-Host "Python processes running: $($pythonProcs.Count)" -ForegroundColor Green
    foreach ($proc in $pythonProcs) {
        $runtime = (Get-Date) - $proc.StartTime
        Write-Host "  PID $($proc.Id): Running for $([math]::Round($runtime.TotalMinutes, 1)) minutes" -ForegroundColor Gray
    }
} else {
    Write-Host "No Python processes running" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Old Files Check ===" -ForegroundColor Cyan
Write-Host "Checking for files older than today..." -ForegroundColor Gray

$today = (Get-Date).Date
foreach ($comp in $competitions.Keys | Sort-Object) {
    $path = "data\race_segments\$comp"
    if (Test-Path $path) {
        $oldFiles = Get-ChildItem -Path $path -Filter "*.mp4" -File | Where-Object { $_.CreationTime.Date -lt $today }
        if ($oldFiles) {
            $oldCount = ($oldFiles | Measure-Object).Count
            Write-Host "$comp`: $oldCount old files (may need regeneration)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "Run this script anytime to check progress!" -ForegroundColor Cyan
