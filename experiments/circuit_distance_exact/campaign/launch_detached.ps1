# Start certify_shift.sh OUTSIDE every Claude session and outside WMI, as an on-demand Scheduled Task: its root's
# parent is the Schedule service (svchost), so neither the session ending nor any tree- or job-kill of WmiPrvSE can
# reach it. History: a job started from a tool shell dies with the session (2026-10-21, a lost cell); a WMI-created
# job lives in WmiPrvSE's job object -- 2026-10-22 23:27 that job died and took this campaign and another session's
# 13 h gate with it, and orphaning via `cmd /c start` did not help, because job membership is inherited.
# Bash is named by its absolute path (a task on a bare Store-alias executable never launches).
#   powershell -File launch_detached.ps1 [-Threads 20] [-Priority BelowNormal]
# Confirm with `Get-ScheduledTask ONE_Q-R198-certify-shift` and the START line in out/campaign.log.
param([int]$Threads = 20, [ValidateSet('Normal', 'BelowNormal', 'Idle')][string]$Priority = 'Normal')
$camp = Split-Path -Parent $MyInvocation.MyCommand.Path
$bash = Join-Path ${env:ProgramFiles} "Git\bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found at $bash" }
$script = (Join-Path $camp "certify_shift.sh") -replace '\\', '/'
$out = (Join-Path $camp "out") -replace '\\', '/'
New-Item -ItemType Directory -Force -Path (Join-Path $camp "out") | Out-Null
$name = 'ONE_Q-R198-certify-shift'
# Task Scheduler priority: 4-6 normal, 7-8 below normal, 9-10 idle
$prio = @{ Normal = 5; BelowNormal = 7; Idle = 9 }[$Priority]
$action = New-ScheduledTaskAction -Execute $bash -Argument "-l -c `"THREADS=$Threads bash '$script' >> '$out/runner.out' 2>&1`"" -WorkingDirectory $camp
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Days 3) -Priority $prio -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew
if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) { Unregister-ScheduledTask -TaskName $name -Confirm:$false }
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Description 'ONE_Q R198: 3-colour time-shift certificates (resumable)' | Out-Null
Start-ScheduledTask -TaskName $name
"started scheduled task $name ($Threads threads, task priority $prio)"
