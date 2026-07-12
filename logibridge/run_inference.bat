@echo off
setlocal EnableExtensions

set "MODE=%~1"
if not defined MODE set "MODE=demo"

set "ANOMALY=%~2"
if not defined ANOMALY set "ANOMALY=combined"

set "TRUCK_ID=%~3"
if not defined TRUCK_ID set "TRUCK_ID=TRUCK_001"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "REPO_ROOT=%SCRIPT_DIR%"
for %%I in ("%REPO_ROOT%\..") do set "ASSIGNMENT_ROOT=%%~fI"
set "PYTHON_EXE=%ASSIGNMENT_ROOT%\.venv312\Scripts\python.exe"
set "BROKER_HOST=localhost"
set "BROKER_PORT=1883"
set "DOCKER_DESKTOP=%ProgramFiles%\Docker\Docker\Docker Desktop.exe"

if /I "%MODE%"=="component-inference" goto :component_inference
if /I "%MODE%"=="component-monitor" goto :component_monitor
if /I "%MODE%"=="component-simulator" goto :component_simulator
if /I "%MODE%"=="inference" goto :run_inference_only
if /I "%MODE%"=="monitor" goto :run_monitor_only
if /I "%MODE%"=="simulator" goto :run_simulator_only
if /I "%MODE%"=="demo" goto :run_demo
if /I "%MODE%"=="help" goto :usage_ok

echo [ERROR] Unsupported mode: %MODE%
goto :usage_error

:run_demo
call :verify_python || exit /b %errorlevel%
call :verify_demo_files || exit /b %errorlevel%
call :ensure_broker || exit /b %errorlevel%

echo [INFO] Launching LogiBridge demo for truck %TRUCK_ID% with anomaly %ANOMALY%.
start "LogiBridge Inference" cmd /k ""%~f0" component-inference placeholder %TRUCK_ID%"
start "LogiBridge Drift Monitor" cmd /k ""%~f0" component-monitor placeholder %TRUCK_ID%"
timeout /t 2 /nobreak >nul
start "LogiBridge Simulator" cmd /k ""%~f0" component-simulator %ANOMALY% %TRUCK_ID%"

echo [INFO] Demo windows opened: broker, inference, drift monitor, simulator.
echo [INFO] Close the opened windows to stop the demo.
exit /b 0

:run_inference_only
call :verify_python || exit /b %errorlevel%
call :verify_demo_files || exit /b %errorlevel%
call :ensure_broker || exit /b %errorlevel%
goto :component_inference

:run_monitor_only
call :verify_python || exit /b %errorlevel%
call :verify_demo_files || exit /b %errorlevel%
call :ensure_broker || exit /b %errorlevel%
goto :component_monitor

:run_simulator_only
call :verify_python || exit /b %errorlevel%
call :ensure_broker || exit /b %errorlevel%
goto :component_simulator

:component_inference
call :verify_python || exit /b %errorlevel%
set "TRUCK_ID=%~3"
if not defined TRUCK_ID set "TRUCK_ID=TRUCK_001"
cd /d "%REPO_ROOT%"
set "MQTT_BROKER_HOST=%BROKER_HOST%"
set "MQTT_BROKER_PORT=%BROKER_PORT%"
set "TRUCK_ID=%TRUCK_ID%"
echo [INFO] Starting inference service for %TRUCK_ID%.
"%PYTHON_EXE%" -u inference\inference_service.py
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Inference service exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:component_monitor
call :verify_python || exit /b %errorlevel%
set "TRUCK_ID=%~3"
if not defined TRUCK_ID set "TRUCK_ID=TRUCK_001"
cd /d "%REPO_ROOT%"
set "MQTT_BROKER_HOST=%BROKER_HOST%"
set "MQTT_BROKER_PORT=%BROKER_PORT%"
set "TRUCK_ID=%TRUCK_ID%"
echo [INFO] Starting drift monitor for %TRUCK_ID%.
"%PYTHON_EXE%" -u monitoring\drift_monitor.py --truck-id "%TRUCK_ID%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Drift monitor exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:component_simulator
call :verify_python || exit /b %errorlevel%
set "ANOMALY=%~2"
if not defined ANOMALY set "ANOMALY=combined"
set "TRUCK_ID=%~3"
if not defined TRUCK_ID set "TRUCK_ID=TRUCK_001"
cd /d "%REPO_ROOT%"
echo [INFO] Starting simulator for %TRUCK_ID% with anomaly %ANOMALY%.
"%PYTHON_EXE%" -u data_pipeline\simulator.py --anomaly "%ANOMALY%" --truck-id "%TRUCK_ID%"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] Simulator exited with code %EXIT_CODE%.
    pause
)
exit /b %EXIT_CODE%

:verify_python
if exist "%PYTHON_EXE%" exit /b 0
echo [ERROR] Missing interpreter: %PYTHON_EXE%
echo [ERROR] This demo expects the workspace virtual environment at ..\.venv312.
exit /b 2

:verify_demo_files
for %%F in (
    "%REPO_ROOT%\inference\inference_service.py"
    "%REPO_ROOT%\monitoring\drift_monitor.py"
    "%REPO_ROOT%\data_pipeline\simulator.py"
    "%REPO_ROOT%\training\models\model_int8.tflite"
    "%REPO_ROOT%\data_pipeline\training_stats.npy"
) do (
    if not exist "%%~F" (
        echo [ERROR] Missing required file: %%~F
        exit /b 3
    )
)
exit /b 0

:ensure_broker
call :broker_is_up
if "%ERRORLEVEL%"=="0" (
    echo [INFO] MQTT broker is already listening on %BROKER_HOST%:%BROKER_PORT%.
    exit /b 0
)

where mosquitto.exe >nul 2>&1
if "%ERRORLEVEL%"=="0" (
    echo [INFO] Starting local Mosquitto broker.
    start "LogiBridge MQTT Broker" cmd /k "mosquitto -v"
    call :wait_for_broker && exit /b 0
    echo [ERROR] Mosquitto did not open %BROKER_HOST%:%BROKER_PORT%.
    exit /b 4
)

call :ensure_docker_daemon
if "%ERRORLEVEL%"=="0" (
    docker ps -a --format "{{.Names}}" | findstr /I /X "logibridge-mqtt" >nul 2>&1
    if "%ERRORLEVEL%"=="0" (
        echo [INFO] Starting existing Docker MQTT container.
        docker start logibridge-mqtt >nul
    ) else (
        echo [INFO] Creating Docker MQTT container.
        docker run -d --name logibridge-mqtt -p %BROKER_PORT%:%BROKER_PORT% eclipse-mosquitto >nul
    )
    call :wait_for_broker && exit /b 0
    echo [ERROR] Docker broker did not open %BROKER_HOST%:%BROKER_PORT%.
    exit /b 5
)

echo [ERROR] No MQTT broker is available on %BROKER_HOST%:%BROKER_PORT%.
echo [ERROR] Install Mosquitto or start Docker Desktop, then rerun this file.
exit /b 6

:ensure_docker_daemon
docker info >nul 2>&1
if "%ERRORLEVEL%"=="0" exit /b 0

if exist "%DOCKER_DESKTOP%" (
    echo [INFO] Starting Docker Desktop.
    start "Docker Desktop" "%DOCKER_DESKTOP%"
    for /L %%I in (1,1,45) do (
        docker info >nul 2>&1
        if not errorlevel 1 exit /b 0
        timeout /t 2 /nobreak >nul
    )
)

exit /b 1

:broker_is_up
powershell -NoProfile -Command "$client = New-Object Net.Sockets.TcpClient; try { $client.Connect('%BROKER_HOST%', %BROKER_PORT%); exit 0 } catch { exit 1 } finally { if ($client.Connected) { $client.Close() } }" >nul 2>&1
exit /b %ERRORLEVEL%

:wait_for_broker
for /L %%I in (1,1,20) do (
    call :broker_is_up
    if not errorlevel 1 (
        echo [INFO] MQTT broker is ready on %BROKER_HOST%:%BROKER_PORT%.
        exit /b 0
    )
    timeout /t 1 /nobreak >nul
)
exit /b 1

:usage_ok
echo Usage:
echo   run_inference.bat
echo   run_inference.bat demo [anomaly] [truck_id]
echo   run_inference.bat inference [ignored] [truck_id]
echo   run_inference.bat monitor [ignored] [truck_id]
echo   run_inference.bat simulator [anomaly] [truck_id]
echo.
echo Default mode is demo, which starts broker, inference, drift monitor, and simulator.
exit /b 0

:usage_error
call :usage_ok
exit /b 64