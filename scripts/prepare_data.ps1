param(
    [string]$ZipPath = "C:\Users\user\Downloads\disaster_label_splits_70_15_15.zip",
    [string]$DataDir = ".\data"
)

New-Item -ItemType Directory -Force $DataDir | Out-Null
tar -xf $ZipPath -C $DataDir
Write-Host "Extracted dataset to $DataDir"
