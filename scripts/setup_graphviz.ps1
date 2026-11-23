# Install the portable renderer used by lesson 07 and the birds slides.
$ErrorActionPreference = 'Stop'
$projectVenv = Join-Path (Split-Path $PSScriptRoot) '.venv'
if (-not (Test-Path (Join-Path $projectVenv 'pyvenv.cfg'))) {
    throw 'Create the course .venv before installing the renderer.'
}
$toolsDir = Join-Path $projectVenv 'course-tools'
$dotExe = Join-Path $toolsDir 'Graphviz-14.1.5-win64\bin\dot.exe'
if (-not (Test-Path -LiteralPath $dotExe)) {
    New-Item -ItemType Directory -Path $toolsDir -Force | Out-Null
    $archive = Join-Path $toolsDir 'Graphviz-14.1.5-win64.zip'
    $url = 'https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/14.1.5/windows_10_cmake_Release_Graphviz-14.1.5-win64.zip'
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $archive
    if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne 'b18235a3f4291c58b01055f683548cc8e433d81ac880c8660a2a9dc5e21ca012') {
        throw 'The downloaded Graphviz archive did not match the release checksum.'
    }
    Expand-Archive -LiteralPath $archive -DestinationPath $toolsDir -Force
}
& $dotExe -V
