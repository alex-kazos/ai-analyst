@echo off
setlocal enabledelayedexpansion

:: Colors
set "GREEN=[32m"
set "YELLOW=[33m"
set "NC=[0m"

:: Check if .env exists
if not exist .env (
    echo %YELLOW%Warning: .env file not found. Creating from .env.example...%NC%
    copy .env.example .env >nul
    echo %GREEN%Created .env file. Please edit it with your configuration.%NC%
    echo.
)

:: Load environment variables
for /f "usebackq tokens=*" %%i in (`.env`) do (
    set "%%i"
)

:: Main command processing
if "%~1"=="" goto show_help

if "%~1"=="up" (
    echo %GREEN%Starting services...%NC%
    docker-compose up -d
    goto :eof
)

if "%~1"=="down" (
    echo %YELLOW%Stopping services...%NC%
    docker-compose down
    goto :eof
)

if "%~1"=="build" (
    echo %GREEN%Building services...%NC%
    docker-compose build --no-cache
    goto :eof
)

if "%~1"=="logs" (
    docker-compose logs -f
    goto :eof
)

if "%~1"=="shell" (
    docker-compose exec web bash
    goto :eof
)

if "%~1"=="shell-db" (
    docker-compose exec db psql -U %DB_USER% -d %DB_NAME%
    goto :eof
)

if "%~1"=="shell-redis" (
    docker-compose exec redis redis-cli -a "%REDIS_PASSWORD%"
    goto :eof
)

if "%~1"=="migrate" (
    echo %GREEN%Running migrations...%NC%
    docker-compose exec web python manage.py migrate
    goto :eof
)

if "%~1"=="createsuperuser" (
    docker-compose exec web python manage.py createsuperuser
    goto :eof
)

if "%~1"=="collectstatic" (
    echo %GREEN%Collecting static files...%NC%
    docker-compose exec web python manage.py collectstatic --noinput
    goto :eof
)

if "%~1"=="test" (
    echo %GREEN%Running tests...%NC%
    docker-compose exec web python manage.py test
    goto :eof
)

if "%~1"=="help" goto show_help
if "%~1"=="--help" goto show_help
if "%~1"=="-h" goto show_help

echo %YELLOW%Unknown command: %~1%NC%
echo.

:show_help
echo Usage: %~nx0 [command]
echo.
echo Available commands:
echo   up           Start all services in detached mode
echo   down         Stop and remove all services
echo   build        Build or rebuild services
echo   logs         View output from containers
echo   shell        Open a bash shell in the web container
echo   shell-db     Open a psql shell in the database container
echo   shell-redis  Open a redis-cli shell in the redis container
echo   migrate      Run database migrations
echo   createsuperuser  Create a superuser
echo   collectstatic  Collect static files
echo   test        Run tests
echo   help         Show this help message
echo.

goto :eof
