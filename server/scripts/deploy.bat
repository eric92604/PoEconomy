@echo off
REM PoEconomy Live Dashboard Deployment Script for Windows
REM This script automates the deployment of your investment dashboard to Cloudflare

setlocal enabledelayedexpansion

echo.
echo ^🚀 PoEconomy Live Dashboard Deployment
echo ====================================

REM Configuration
set PROJECT_NAME=poeconomy-dashboard
set WORKER_NAME=poeconomy-api
set DOMAIN=
set NODE_ENV=production

REM Check prerequisites
echo Checking prerequisites...

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Node.js is not installed. Please install Node.js first.
    pause
    exit /b 1
)

where npm >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ npm is not installed. Please install npm first.
    pause
    exit /b 1
)

where wrangler >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠ Wrangler CLI not found. Installing...
    npm install -g wrangler
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ✗ Python is not installed. Please install Python first.
    pause
    exit /b 1
)

echo ✓ All prerequisites satisfied

REM Get domain from user
echo.
set /p DOMAIN="Enter your domain name (e.g., yourdomain.com): "

if "!DOMAIN!"=="" (
    echo ✗ Domain name is required
    pause
    exit /b 1
)

echo.
echo Configuration:
echo - Project Name: %PROJECT_NAME%
echo - Worker Name: %WORKER_NAME%
echo - Domain: !DOMAIN!
echo - Environment: %NODE_ENV%
echo.

set /p CONTINUE="Continue with deployment? (y/N): "
if /i not "!CONTINUE!"=="y" (
    echo Deployment cancelled.
    pause
    exit /b 0
)

REM Create deployment directory
set DEPLOY_DIR=%CD%\deploy-temp
mkdir "%DEPLOY_DIR%" 2>nul

echo ✓ Created deployment directory: %DEPLOY_DIR%

REM Step 1: Prepare frontend files
echo.
echo 📁 Step 1: Preparing frontend files...

copy ..\public\index.html "%DEPLOY_DIR%\" >nul

REM Update API URLs in the HTML file (basic replacement)
powershell -Command "(Get-Content '%DEPLOY_DIR%\index.html') -replace 'http://localhost:3000/api', 'https://api.!DOMAIN!/api' -replace \"'/api'\", \"'https://api.!DOMAIN!/api'\" | Set-Content '%DEPLOY_DIR%\index.html'"

echo ✓ Frontend files prepared

REM Step 2: Deploy frontend to Cloudflare Pages
echo.
echo 🌐 Step 2: Deploying frontend to Cloudflare Pages...

cd /d "%DEPLOY_DIR%"

REM Check if wrangler is logged in
wrangler whoami >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠ Please log in to Cloudflare first:
    wrangler login
)

REM Deploy to Pages
echo Deploying to Cloudflare Pages...
wrangler pages deploy . --project-name="%PROJECT_NAME%"

if %errorlevel% neq 0 (
    echo ✗ Pages deployment failed
    pause
    exit /b 1
)

echo ✓ Pages deployment successful

REM Step 3: Configure custom domain
echo.
echo 🔗 Step 3: Configuring custom domain...
echo ⚠ Please manually configure the custom domain in Cloudflare Pages:
echo   1. Go to Cloudflare Dashboard ^> Pages ^> %PROJECT_NAME%
echo   2. Go to Custom domains tab
echo   3. Add domain: dashboard.!DOMAIN!

REM Step 4: Prepare and deploy Worker
echo.
echo ⚡ Step 4: Deploying API Worker...

cd /d "%~dp0"

REM Create worker directory
set WORKER_DIR=%CD%\worker-temp
mkdir "%WORKER_DIR%" 2>nul

REM Copy worker files
copy ..\workers\api.js "%WORKER_DIR%\worker.js" >nul

REM Create wrangler.toml
(
echo name = "%WORKER_NAME%"
echo main = "worker.js"
echo compatibility_date = "2024-01-01"
echo.
echo [env.production]
echo name = "%WORKER_NAME%"
echo route = "api.!DOMAIN!/*"
echo.
echo # Environment variables
echo [env.production.vars]
echo NODE_ENV = "%NODE_ENV%"
) > "%WORKER_DIR%\wrangler.toml"

REM Update worker.js with actual domain
powershell -Command "(Get-Content '%WORKER_DIR%\worker.js') -replace 'https://your-domain.com/dashboard', 'https://dashboard.!DOMAIN!' | Set-Content '%WORKER_DIR%\worker.js'"

cd /d "%WORKER_DIR%"

REM Deploy worker
echo Deploying Worker to Cloudflare...
wrangler deploy --env production

if %errorlevel% neq 0 (
    echo ✗ Worker deployment failed
    pause
    exit /b 1
)

echo ✓ API Worker deployed

REM Step 5: DNS Configuration
echo.
echo 🌍 Step 5: DNS Configuration...
echo ⚠ Please ensure the following DNS records are configured:
echo   1. dashboard.!DOMAIN! -^> Cloudflare Pages (should be automatic)
echo   2. api.!DOMAIN! -^> Worker route (should be automatic)

REM Step 6: Test deployment
echo.
echo 🧪 Step 6: Testing deployment...

set API_URL=https://api.!DOMAIN!/api/health
set DASHBOARD_URL=https://dashboard.!DOMAIN!

echo ✓ API URL: !API_URL!
echo ✓ Dashboard URL: !DASHBOARD_URL!

echo.
echo Testing API endpoint...
curl -f -s "!API_URL!" >nul 2>&1
if %errorlevel% eq 0 (
    echo ✓ API endpoint is responding
) else (
    echo ⚠ API endpoint not yet available (DNS propagation may take a few minutes)
)

REM Step 7: Python Backend Integration
echo.
echo 🐍 Step 7: Python Backend Integration...

set PYTHON_SCRIPT=..\..\ml\scripts\generate_dashboard_data.py
if exist "%PYTHON_SCRIPT%" (
    echo ✓ Python script found: %PYTHON_SCRIPT%
    
    REM Test script execution
    python "%PYTHON_SCRIPT%" --stdout >nul 2>&1
    if %errorlevel% eq 0 (
        echo ✓ Python script executed successfully
    ) else (
        echo ⚠ Python script execution failed. Check dependencies.
    )
) else (
    echo ⚠ Python script not found: %PYTHON_SCRIPT%
)

REM Step 8: Cleanup
echo.
echo 🧹 Step 8: Cleanup...

cd /d "%~dp0"
rmdir /s /q "%DEPLOY_DIR%" 2>nul
rmdir /s /q "%WORKER_DIR%" 2>nul

echo ✓ Temporary files cleaned up

REM Final instructions
echo.
echo 🎉 Deployment Complete!
echo =====================
echo.
echo Your live investment dashboard has been deployed:
echo.
echo 📊 Dashboard: https://dashboard.!DOMAIN!
echo 🔌 API: https://api.!DOMAIN!
echo.
echo Next Steps:
echo 1. Wait for DNS propagation (up to 24 hours, usually much faster)
echo 2. Configure your database connection in the Worker environment variables
echo 3. Set up scheduled data updates (optional)
echo 4. Configure SSL/TLS settings in Cloudflare dashboard
echo.
echo Monitoring:
echo - Cloudflare Analytics: Monitor traffic and performance
echo - Worker Logs: Check API performance and errors
echo - Pages Analytics: Track dashboard usage
echo.
echo Need help? Check the deployment guide: ..\docs\deployment_guide.md
echo.

REM Optional: Open dashboard in browser
set /p OPEN_BROWSER="Open dashboard in browser? (y/N): "
if /i "!OPEN_BROWSER!"=="y" (
    start https://dashboard.!DOMAIN!
)

echo.
echo ✓ Deployment script completed successfully!
pause 