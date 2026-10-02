<# : batch portion
:: SatoDive ComfyUI Doctor - double-click this file. It checks your ComfyUI install, fixes what is safe, and writes a report.
@echo off
setlocal
title SatoDive ComfyUI Doctor
set "SATO_SELF=%~f0"
set "SATO_COMFY=%~1"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:SATO_SELF); Invoke-Expression $s"
echo.
pause
exit /b
#>

# =====================================================================================
#  SatoDive ComfyUI Doctor
#  Checks your ComfyUI install, fixes what is safe to fix, and writes a report.
#  Nothing is ever deleted: anything it removes is MOVED to ComfyUI\_satodive_backup\<date>
# =====================================================================================

$ErrorActionPreference = 'Continue'
$Report = New-Object System.Collections.Generic.List[string]
$Fixes = 0; $Problems = 0
$Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$PackFolderName = 'ComfyUI-LoRA-Merge-SatoDive'
$PackFiles = @('__init__.py', 'sato_nodes.py', 'sato_arch.py', 'sato_lora_core.py', 'sato_preview.py',
               'sato_routes.py', 'sato_civitai.py', 'sato_help.py', 'web\js\satodive_lora_merge.js')
$ObsoleteFiles = @('sato_nodes_v2.py')

function Say([string]$msg, [string]$kind = 'INFO') {
    $line = '[{0,-5}] {1}' -f $kind, $msg
    $Report.Add($line)
    $color = @{ OK = 'Green'; FIX = 'Cyan'; WARN = 'Yellow'; ERROR = 'Red'; INFO = 'Gray'; STEP = 'White' }[$kind]
    if (-not $color) { $color = 'Gray' }
    Write-Host $line -ForegroundColor $color
    if ($kind -eq 'FIX') { $script:Fixes++ }
    if ($kind -eq 'WARN' -or $kind -eq 'ERROR') { $script:Problems++ }
}
function Step([string]$title) {
    $Report.Add(''); Write-Host ''
    Say ('==== ' + $title + ' ====') 'STEP'
}
function Ask([string]$question) {
    $a = Read-Host ($question + ' [Y/N]')
    return ($a -match '^\s*(y|yes|o|oui)\s*$')
}
function Backup-Dir {
    $d = Join-Path $script:Comfy ('_satodive_backup\' + $script:Stamp)
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
    return $d
}
function Move-ToBackup([string]$path) {
    $dest = Join-Path (Backup-Dir) (Split-Path $path -Leaf)
    $i = 1
    while (Test-Path $dest) { $dest = Join-Path (Backup-Dir) ((Split-Path $path -Leaf) + '_' + $i); $i++ }
    Move-Item -LiteralPath $path -Destination $dest -Force -ErrorAction Stop
    return $dest
}
function Pack-Version([string]$dir) {
    $pp = Join-Path $dir 'pyproject.toml'
    if (Test-Path $pp) {
        $m = Select-String -Path $pp -Pattern '^\s*version\s*=\s*"([^"]+)"' | Select-Object -First 1
        if ($m) { return $m.Matches[0].Groups[1].Value }
    }
    return '0'
}
function Version-Key([string]$v) {
    try { return [version]($v -replace '[^0-9.]', '') } catch { return [version]'0.0' }
}

Write-Host ''
Write-Host '  SatoDive ComfyUI Doctor' -ForegroundColor Magenta
Write-Host '  checks + safe fixes + report' -ForegroundColor DarkGray

# -------------------------------------------------------------------------------------
Step 'ComfyUI folder'
# -------------------------------------------------------------------------------------
$Here = Split-Path -Parent $env:SATO_SELF
$candidates = @($env:SATO_COMFY, $Here, (Join-Path $Here 'ComfyUI'), (Join-Path $Here 'ComfyUI-Easy-Install\ComfyUI'),
                (Split-Path -Parent $Here), (Join-Path (Split-Path -Parent $Here) 'ComfyUI'),
                'C:\Ai\ComfyUI-Easy-Install\ComfyUI-Easy-Install\ComfyUI') | Where-Object { $_ }
$Comfy = $null
foreach ($c in $candidates) {
    if ((Test-Path (Join-Path $c 'main.py')) -and (Test-Path (Join-Path $c 'custom_nodes'))) { $Comfy = (Resolve-Path $c).Path; break }
}
while (-not $Comfy) {
    $p = Read-Host 'Could not find ComfyUI. Paste the path of your ComfyUI folder (the one with main.py), or leave empty to quit'
    if (-not $p) { Say 'No ComfyUI folder given - nothing checked.' 'ERROR'; break }
    $p = $p.Trim('"', ' ')
    if ((Test-Path (Join-Path $p 'main.py')) -and (Test-Path (Join-Path $p 'custom_nodes'))) { $Comfy = (Resolve-Path $p).Path }
    else { Write-Host 'That folder has no main.py / custom_nodes, try again.' -ForegroundColor Yellow }
}

if ($Comfy) {
    Say ('ComfyUI found: ' + $Comfy) 'OK'
    $CustomNodes = Join-Path $Comfy 'custom_nodes'
    $Python = $null
    foreach ($py in @((Join-Path (Split-Path -Parent $Comfy) 'python_embeded\python.exe'), (Join-Path $Comfy 'venv\Scripts\python.exe'),
                      (Join-Path $Comfy '.venv\Scripts\python.exe'))) {
        if (Test-Path $py) { $Python = $py; break }
    }
    if ($Python) { Say ('Python: ' + $Python) 'OK' } else { Say 'Embedded Python not found (file compile check will be skipped).' 'INFO' }

    # ---------------------------------------------------------------------------------
    Step 'Port 8188 (the ComfyUI web server)'
    # ---------------------------------------------------------------------------------
    $port = 8188
    $listen = @()
    try { $listen = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop | Select-Object -ExpandProperty OwningProcess -Unique) } catch { }
    if ($listen.Count -eq 0) {
        Say 'Port 8188 is free.' 'OK'
    } else {
        foreach ($procId in $listen) {
            $proc = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $procId) -ErrorAction SilentlyContinue
            $name = if ($proc) { $proc.Name } else { 'unknown' }
            $cmd = if ($proc -and $proc.CommandLine) { $proc.CommandLine } else { '' }
            Say ('Port 8188 is used by ' + $name + ' (PID ' + $procId + ').') 'WARN'
            if ($cmd) { Say ('   command: ' + $cmd) 'INFO' }
            if ($cmd -match 'main\.py') {
                Say '   This is a ComfyUI that is still running (an old window / background process).' 'INFO'
            }
            if (Ask ('Close ' + $name + ' (PID ' + $procId + ') so ComfyUI can start?')) {
                try { Stop-Process -Id $procId -Force -ErrorAction Stop; Start-Sleep -Seconds 1; Say ('Closed ' + $name + ' (PID ' + $procId + ').') 'FIX' }
                catch { Say ('Could not close PID ' + $procId + ': ' + $_.Exception.Message + ' (try running this file as administrator).') 'ERROR' }
            } else {
                Say 'Left it running. Alternative: add  --port 8189  to your ComfyUI start line.' 'INFO'
            }
        }
    }
    # other ComfyUI processes that are still alive but not listening (stuck)
    $stuck = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
               Where-Object { $_.CommandLine -match 'main\.py' -and $listen -notcontains $_.ProcessId })
    foreach ($s in $stuck) {
        Say ('Another ComfyUI process is still alive: PID ' + $s.ProcessId) 'WARN'
        if (Ask ('Close it (PID ' + $s.ProcessId + ')?')) {
            try { Stop-Process -Id $s.ProcessId -Force -ErrorAction Stop; Say ('Closed PID ' + $s.ProcessId + '.') 'FIX' }
            catch { Say ('Could not close PID ' + $s.ProcessId + ': ' + $_.Exception.Message) 'ERROR' }
        }
    }
    # Windows reserved port ranges (Hyper-V / WSL / Docker)
    $reserved = $false
    $ranges = netsh interface ipv4 show excludedportrange protocol=tcp 2>$null
    foreach ($line in $ranges) {
        if ($line -match '^\s*(\d+)\s+(\d+)') {
            if ($port -ge [int]$Matches[1] -and $port -le [int]$Matches[2]) { $reserved = $true; Say ('Windows reserves ports ' + $Matches[1] + '-' + $Matches[2] + ' (includes 8188).') 'WARN' }
        }
    }
    if (-not $reserved) { Say 'Port 8188 is not reserved by Windows.' 'OK' }
    else { Say 'Fix: in an ADMIN command prompt run  net stop winnat  then  net start winnat  - or start ComfyUI with --port 8189.' 'INFO' }

    # ---------------------------------------------------------------------------------
    Step 'LoRA Merge Studio - SatoDive (install location)'
    # ---------------------------------------------------------------------------------
    $fixesBeforePack = $Fixes
    $copies = @(Get-ChildItem -LiteralPath $CustomNodes -Recurse -Depth 3 -Filter 'sato_lora_core.py' -File -ErrorAction SilentlyContinue |
                ForEach-Object { $_.Directory.FullName } | Select-Object -Unique)
    $zips = @(Get-ChildItem -LiteralPath $CustomNodes -Filter '*SatoDive*.zip' -File -ErrorAction SilentlyContinue)
    $Pack = $null
    if ($copies.Count -eq 0) {
        Say 'The LoRA Merge pack is NOT installed in custom_nodes.' 'WARN'
        if ($zips.Count -gt 0) {
            $zip = $zips | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if (Ask ('Found ' + $zip.Name + ' in custom_nodes. Extract it correctly now?')) {
                $tmp = Join-Path ([IO.Path]::GetTempPath()) ('satodive_' + $Stamp)
                Expand-Archive -LiteralPath $zip.FullName -DestinationPath $tmp -Force
                $inner = Get-ChildItem -LiteralPath $tmp -Recurse -Filter 'sato_lora_core.py' -File | Select-Object -First 1
                if ($inner) {
                    Move-Item -LiteralPath $inner.Directory.FullName -Destination (Join-Path $CustomNodes $PackFolderName) -Force
                    $Pack = Join-Path $CustomNodes $PackFolderName
                    Say ('Extracted to ' + $Pack) 'FIX'
                } else { Say 'The zip does not contain the pack files.' 'ERROR' }
            }
        } else {
            Say ('Unzip "ComfyUI-LoRA-Merge - SatoDive.zip" so that you get  ' + $CustomNodes + '\' + $PackFolderName + '\__init__.py') 'INFO'
        }
    } else {
        foreach ($c in $copies) { Say ('Found a copy: ' + $c + '  (version ' + (Pack-Version $c) + ')') 'INFO' }
        # keep the newest version (then the newest files)
        $Pack = $copies | Sort-Object @{ Expression = { Version-Key (Pack-Version $_) }; Descending = $true },
                                      @{ Expression = { (Get-Item (Join-Path $_ 'sato_lora_core.py')).LastWriteTime }; Descending = $true } |
                Select-Object -First 1
        # 1) the copy we keep must sit DIRECTLY in custom_nodes - pull it out first, so moving
        #    other copies / wrapper folders to backup can never take it along
        $wrapper = $null
        if ((Split-Path -Parent $Pack) -ne $CustomNodes) {
            $wrapper = Split-Path -Parent $Pack
            while ((Split-Path -Parent $wrapper) -and (Split-Path -Parent $wrapper) -ne $CustomNodes) { $wrapper = Split-Path -Parent $wrapper }
            Say ('The pack is one folder too deep - ComfyUI cannot see it: ' + $Pack) 'WARN'
            $tmp = Join-Path $CustomNodes ('_satodive_new_' + $Stamp)
            try {
                Move-Item -LiteralPath $Pack -Destination $tmp -Force -ErrorAction Stop
                $Pack = $tmp
                Say 'Pulled the pack out to the custom_nodes level.' 'FIX'
            } catch {
                Say ('Could not move the pack: ' + $_.Exception.Message + ' (close ComfyUI and run again).') 'ERROR'
                $wrapper = $null
            }
        }
        # 2) every other copy goes to backup (ComfyUI would otherwise load several copies)
        $others = @($copies | Where-Object { $_ -ne $Pack -and (Test-Path (Join-Path $_ 'sato_lora_core.py')) })
        if ($others.Count -gt 0) {
            Say ('Found ' + ($others.Count + 1) + ' copies - keeping only the newest one.') 'WARN'
            foreach ($c in $others) {
                try { $d = Move-ToBackup $c; Say ('Moved an extra copy to backup: ' + $d) 'FIX' }
                catch { Say ('Could not move ' + $c + ': ' + $_.Exception.Message + ' (is ComfyUI still running?)') 'ERROR' }
            }
        }
        # 3) an empty wrapper folder left behind (no python files = not another node pack) goes to backup
        if ($wrapper -and (Test-Path $wrapper) -and -not (Test-Path (Join-Path $wrapper '__init__.py')) -and
            -not (Get-ChildItem -LiteralPath $wrapper -Recurse -Filter '*.py' -File -ErrorAction SilentlyContinue)) {
            try { $d = Move-ToBackup $wrapper; Say ('Moved the empty wrapper folder to backup: ' + $d) 'FIX' }
            catch { Say ('Could not move the wrapper folder ' + $wrapper + ': ' + $_.Exception.Message) 'WARN' }
        }
        # 4) final, clean folder name
        if ((Split-Path -Leaf $Pack) -ne $PackFolderName) {
            $target = Join-Path $CustomNodes $PackFolderName
            try {
                if (Test-Path $target) { $d = Move-ToBackup $target; Say ('Moved an old ' + $PackFolderName + ' folder to backup: ' + $d) 'FIX' }
                Rename-Item -LiteralPath $Pack -NewName $PackFolderName -ErrorAction Stop
                $Pack = $target
                Say ('Pack folder is now: ' + $Pack) 'FIX'
            } catch { Say ('Could not rename ' + $Pack + ': ' + $_.Exception.Message) 'WARN' }
        }
        if ($Fixes -eq $fixesBeforePack) { Say ('Installed in the right place: ' + $Pack) 'OK' }
        if ($zips.Count -gt 0) { Say ('Note: zip file(s) inside custom_nodes are ignored by ComfyUI: ' + (($zips | ForEach-Object Name) -join ', ')) 'INFO' }
    }

    # ---------------------------------------------------------------------------------
    if ($Pack -and (Test-Path $Pack)) {
        Step 'LoRA Merge Studio - SatoDive (files)'
        Say ('Version: ' + (Pack-Version $Pack)) 'INFO'
        $missing = @($PackFiles | Where-Object { -not (Test-Path (Join-Path $Pack $_)) })
        if ($missing.Count -eq 0) { Say 'All pack files are present.' 'OK' }
        else { Say ('Missing files: ' + ($missing -join ', ') + '  -> unzip the pack again.') 'ERROR' }
        foreach ($f in $ObsoleteFiles) {
            $fp = Join-Path $Pack $f
            if (Test-Path $fp) { $d = Move-ToBackup $fp; Say ('Moved leftover file from an older version to backup: ' + $f) 'FIX' }
        }
        $caches = @(Get-ChildItem -LiteralPath $Pack -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue)
        foreach ($cdir in $caches) { Remove-Item -LiteralPath $cdir.FullName -Recurse -Force -ErrorAction SilentlyContinue }
        if ($caches.Count -gt 0) { Say 'Cleared old Python cache (__pycache__) - ComfyUI rebuilds it.' 'FIX' }
        if ($Python) {
            $bad = 0
            foreach ($f in (Get-ChildItem -LiteralPath $Pack -Filter '*.py' -File)) {
                $out = & $Python -X utf8 -c "import ast,sys; ast.parse(open(sys.argv[1], encoding='utf-8').read())" $f.FullName 2>&1
                if ($LASTEXITCODE -ne 0) { $bad++; Say ($f.Name + ' does not compile: ' + ($out | Select-Object -Last 1)) 'ERROR' }
            }
            if ($bad -eq 0) { Say 'All Python files of the pack compile with your Python.' 'OK' }
        }
    }

    # ---------------------------------------------------------------------------------
    Step 'Last ComfyUI start (user\comfyui.log)'
    # ---------------------------------------------------------------------------------
    $logFile = Join-Path $Comfy 'user\comfyui.log'
    if (Test-Path $logFile) {
        $log = Get-Content -LiteralPath $logFile -Encoding UTF8 -ErrorAction SilentlyContinue
        $failed = @($log | Where-Object { $_ -match 'IMPORT FAILED' })
        if ($failed.Count) { foreach ($l in $failed) { Say ('Import failed: ' + $l.Trim()) 'WARN' } } else { Say 'No custom node import failures in the last log.' 'OK' }
        if ($log -match 'ComfyUI-LoRA-Merge') { Say 'The LoRA Merge pack was loaded in the last start.' 'OK' }
        else { Say 'The LoRA Merge pack was NOT loaded in the last start (it should be fixed now if it was misplaced).' 'INFO' }
        if ($log -match 'error while attempting to bind on address') { Say 'Last start failed because port 8188 was busy (see the port section above).' 'WARN' }
    } else {
        Say 'No comfyui.log found yet.' 'INFO'
    }
}

# -------------------------------------------------------------------------------------
Step 'Summary'
# -------------------------------------------------------------------------------------
Say ('Fixes applied: ' + $Fixes + '   Problems found: ' + $Problems) 'INFO'
if ($Comfy -and (Test-Path (Join-Path $Comfy ('_satodive_backup\' + $Stamp)))) {
    Say ('Moved items are in: ' + (Join-Path $Comfy ('_satodive_backup\' + $Stamp)) + '  (delete it when everything works)') 'INFO'
}
Say 'Now start ComfyUI normally.' 'INFO'
$reportPath = Join-Path $Here ('SatoDive_ComfyUI_Report_' + $Stamp + '.txt')
try {
    $header = @('SatoDive ComfyUI Doctor report', (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), '')
    Set-Content -LiteralPath $reportPath -Value ($header + $Report) -Encoding UTF8
    Write-Host ''
    Write-Host ('Report saved: ' + $reportPath) -ForegroundColor Magenta
    Start-Process notepad.exe -ArgumentList ('"' + $reportPath + '"')
} catch { Write-Host ('Could not save the report: ' + $_.Exception.Message) -ForegroundColor Red }
