# docker-up.ps1 - Windows PowerShell script to manage the Docker environment

# Set Error Action Preference
$ErrorActionPreference = "Stop"

# Load environment variables from .env.local if it exists
$envFile = ".env.local"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([^#][^=]+)=(.*)') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim()
            if ($value -match '^"(.*)"$') { $value = $matches[1] }  # Remove quotes if present
            if ($value -match '^\''(.*)\''$') { $value = $matches[1] }  # Remove single quotes if present
            [System.Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

# Set default UID and GID if not set
if (-not $env:UID) { $env:UID = 1000 }
if (-not $env:GID) { $env:GID = 1000 }

# Function to display colored output
function Write-ColorOutput($ForegroundColor) {
    $color = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $Host.UI.RawUI.ForegroundColor = $color
}

# Function to check if Docker is running
function Test-DockerRunning {
    try {
        docker info | Out-Null
        return $true
    } catch {
        return $false
    }
}

# Function to build and start the services
function Start-DockerServices {
    Write-ColorOutput "cyan" "Building and starting Docker containers..."
    
    # Build the images
    docker-compose build --no-cache
    
    # Start the services
    docker-compose up -d
    
    # Wait for the database to be ready
    Write-ColorOutput "yellow" "Waiting for database to be ready..."
    Start-Sleep -Seconds 10
    
    # Run migrations
    Write-ColorOutput "yellow" "Running database migrations..."
    docker-compose exec -T web python manage.py migrate
    
    # Create superuser if needed
    $createSuperuser = Read-Host "Do you want to create a superuser? (y/n)"
    if ($createSuperuser -eq 'y') {
        docker-compose exec web python manage.py createsuperuser
    }
    
    # Show container status
    docker-compose ps
    
    Write-ColorOutput "green" "\nDocker services are up and running!"
    Write-ColorOutput "green" "Access the application at: http://localhost:8000"
    Write-ColorOutput "green" "Access the admin interface at: http://localhost:8000/admin"
}

# Function to stop the services
function Stop-DockerServices {
    Write-ColorOutput "yellow" "Stopping Docker containers..."
    docker-compose down
}

# Function to view logs
function Show-Logs {
    param (
        [string]$service = ""
    )
    
    if ($service) {
        docker-compose logs -f $service
    } else {
        docker-compose logs -f
    }
}

# Function to open a shell in the web container
function Enter-WebShell {
    docker-compose exec web bash
}

# Function to open a shell in the database container
function Enter-DbShell {
    docker-compose exec db psql -U $env:DB_USER -d $env:DB_NAME
}

# Function to open a shell in the Redis container
function Enter-RedisShell {
    docker-compose exec redis redis-cli -a $env:REDIS_PASSWORD
}

# Main menu
function Show-MainMenu {
    Clear-Host
    Write-ColorOutput "cyan" "=== AI Analyst Docker Management ==="
    Write-ColorOutput "yellow" "1. Start services"
    Write-ColorOutput "yellow" "2. Stop services"
    Write-ColorOutput "yellow" "3. View logs"
    Write-ColorOutput "yellow" "4. Web container shell"
    Write-ColorOutput "yellow" "5. Database shell"
    Write-ColorOutput "yellow" "6. Redis shell"
    Write-ColorOutput "yellow" "7. Run migrations"
    Write-ColorOutput "yellow" "8. Create superuser"
    Write-ColorOutput "red"   "0. Exit"
    
    $choice = Read-Host "Select an option"
    
    switch ($choice) {
        "1" { Start-DockerServices; pause; Show-MainMenu }
        "2" { Stop-DockerServices; pause; Show-MainMenu }
        "3" { Show-Logs; pause; Show-MainMenu }
        "4" { Enter-WebShell; Show-MainMenu }
        "5" { Enter-DbShell; Show-MainMenu }
        "6" { Enter-RedisShell; Show-MainMenu }
        "7" { docker-compose exec web python manage.py migrate; pause; Show-MainMenu }
        "8" { docker-compose exec web python manage.py createsuperuser; pause; Show-MainMenu }
        "0" { exit }
        default { Write-ColorOutput "red" "Invalid option"; pause; Show-MainMenu }
    }
}

# Check if Docker is running
if (-not (Test-DockerRunning)) {
    Write-ColorOutput "red" "Docker is not running. Please start Docker Desktop and try again."
    exit 1
}

# Show the main menu
Show-MainMenu
