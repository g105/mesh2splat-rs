# Download the hair grooms used by the README and examples/groom.rs into assets\.
#
# The models are Cem Yuksel's (https://www.cemyuksel.com/research/hairmodels),
# free for personal and research use. If you publish anything made from them,
# link to that page; acknowledgements are appreciated. They are not in the
# repository for that reason.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\fetch_grooms.ps1 [name ...]
#        (default: straight wCurly)
param([string[]]$Names = @("straight", "wCurly"))
$ErrorActionPreference = "Stop"

$Base = if ($env:M2S_HAIR_BASE) { $env:M2S_HAIR_BASE } else { "https://www.cemyuksel.com/research/hairmodels" }
$Dest = Join-Path (Split-Path $PSScriptRoot -Parent) "assets"
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$Tmp = Join-Path ([IO.Path]::GetTempPath()) ("m2s_hair_" + [Guid]::NewGuid())
New-Item -ItemType Directory -Path $Tmp | Out-Null

function Test-Hair($Path) {
    if (-not (Test-Path $Path)) { return $false }
    $fs = [IO.File]::OpenRead($Path)
    try {
        $b = New-Object byte[] 4
        $n = $fs.Read($b, 0, 4)
        return ($n -eq 4) -and ([Text.Encoding]::ASCII.GetString($b) -eq "HAIR")
    } finally { $fs.Close() }
}

function Get-File($Url, $Out) {
    try { Invoke-WebRequest -Uri $Url -OutFile $Out -UseBasicParsing; return $true }
    catch { return $false }
}

try {
    foreach ($name in $Names) {
        $out = Join-Path $Dest "$name.hair"
        if (Test-Hair $out) { Write-Host "${name}.hair: already there"; continue }
        # The site has offered the models both as plain .hair files and zipped.
        $hair = Join-Path $Tmp "$name.hair"
        $zip = Join-Path $Tmp "$name.zip"
        if ((Get-File "$Base/$name.hair" $hair) -and (Test-Hair $hair)) {
            Move-Item -Force $hair $out
        } elseif (Get-File "$Base/$name.zip" $zip) {
            $dir = Join-Path $Tmp $name
            Expand-Archive -Force $zip $dir
            $found = Get-ChildItem -Recurse -Path $dir -Filter *.hair | Select-Object -First 1
            if (-not $found -or -not (Test-Hair $found.FullName)) { throw "${name}: the zip holds no .hair file" }
            Move-Item -Force $found.FullName $out
        } else {
            throw "${name}: not found at $Base/$name.hair or $Base/$name.zip; download it by hand from $Base into assets\"
        }
        $mb = [math]::Round((Get-Item $out).Length / 1MB, 1)
        Write-Host "${name}.hair: $mb MB"
    }
} finally {
    Remove-Item -Recurse -Force $Tmp -ErrorAction SilentlyContinue
}
