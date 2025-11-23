# Cyberbullying Detection System Startup Script
# This script starts both the backend and frontend services

Write-Host "🚀 Starting Cyberbullying Detection System..." -ForegroundColor Cyan
Write-Host "=======================================" -ForegroundColor Cyan

# Check if Node.js is installed
try {
    $nodeVersion = node --version
    Write-Host "✅ Node.js found: $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js (v16 or higher)" -ForegroundColor Red
    exit 1
}

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "✅ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python (3.8 or higher)" -ForegroundColor Red
    exit 1
}

# Check if backend dependencies are installed
Write-Host "🔍 Checking backend dependencies..." -ForegroundColor Yellow
if (!(Test-Path "Backend\venv") -and !(pip list | Select-String "flask")) {
    Write-Host "📦 Installing backend dependencies..." -ForegroundColor Yellow
    Set-Location Backend
    pip install -r requirements.txt
    Set-Location ..
}

# Check if frontend dependencies are installed
Write-Host "🔍 Checking frontend dependencies..." -ForegroundColor Yellow
if (!(Test-Path "Frontend\node_modules")) {
    Write-Host "📦 Installing frontend dependencies..." -ForegroundColor Yellow
    Set-Location Frontend
    npm install
    Set-Location ..
}

# Check if .env files exist
if (!(Test-Path "Backend\.env")) {
    Write-Host "⚠️  Backend .env file not found. Creating from template..." -ForegroundColor Yellow
    Copy-Item "Backend\.env.example" "Backend\.env"
    Write-Host "📝 Please edit Backend\.env with your OpenAI API key" -ForegroundColor Red
}

Write-Host "" 
Write-Host "🌐 Starting services..." -ForegroundColor Cyan
Write-Host "Backend API will be available at: http://localhost:5000" -ForegroundColor White
Write-Host "Frontend will be available at: http://localhost:3000" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop both services" -ForegroundColor Yellow
Write-Host ""

# Function to cleanup on exit
function Cleanup {
    Write-Host "🛑 Shutting down services..." -ForegroundColor Red
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    exit 0
}

# Set cleanup handler
$null = Register-EngineEvent PowerShell.Exiting -Action { Cleanup }

try {
    # Start backend in background job
    Write-Host "🔧 Starting backend server..." -ForegroundColor Yellow
    $backendJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        Set-Location Backend
        python app.py
    }

    # Wait a moment for backend to start
    Start-Sleep -Seconds 3

    # Start frontend in background job
    Write-Host "🎨 Starting frontend development server..." -ForegroundColor Yellow
    $frontendJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        Set-Location Frontend
        npm start
    }

    Write-Host "✅ Both services are starting up..." -ForegroundColor Green
    Write-Host "⏳ Please wait for the frontend to open in your browser..." -ForegroundColor Cyan

    # Monitor jobs and show output
    while ($true) {
        # Check if backend job failed
        if ($backendJob.State -eq "Failed") {
            Write-Host "❌ Backend failed to start!" -ForegroundColor Red
            Receive-Job $backendJob
            break
        }

        # Check if frontend job failed
        if ($frontendJob.State -eq "Failed") {
            Write-Host "❌ Frontend failed to start!" -ForegroundColor Red
            Receive-Job $frontendJob
            break
        }

        # Show any job output
        Receive-Job $backendJob -Keep | ForEach-Object { Write-Host "[Backend] $_" -ForegroundColor Blue }
        Receive-Job $frontendJob -Keep | ForEach-Object { Write-Host "[Frontend] $_" -ForegroundColor Green }

        Start-Sleep -Seconds 2
    }

} catch {
    Write-Host "❌ Error starting services: $($_.Exception.Message)" -ForegroundColor Red
} finally {
    Cleanup
}
