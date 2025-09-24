# PowerShell version of the run script

param(
    [Parameter(Position=0, Mandatory=$true)]
    [string]$Command
)

if ($Command -eq "install") {
    Write-Output "Installing requirements..."
    pip install -r requirements.txt
    exit 0
} elseif ($Command -eq "test") {
    Write-Output "Running tests..."
    python -m pytest tests/
    exit 0
} else {
    # Assume $Command is a file path
    python src/app.py $Command
    exit 0
}

