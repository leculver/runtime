@echo off
setlocal enabledelayedexpansion

set retry_count=0
set start_time=%time%

:retry

echo Attempt: !retry_count!
timeout /t 2 >nul

call build.cmd %*
if %ERRORLEVEL% neq 0 (
    set /a retry_count+=1
    goto retry
)

set end_time=%time%

echo Completed the build in !retry_count! retries
echo Started at: !start_time!
echo Ended at:   !end_time!

endlocal