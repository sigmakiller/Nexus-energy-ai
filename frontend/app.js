document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const dot = document.getElementById("connection-dot");
    const text = document.getElementById("connection-text");
    const valHVAC = document.getElementById("val-hvac");
    const valLighting = document.getElementById("val-lighting");
    const valMELS = document.getElementById("val-mels");
    const logStream = document.getElementById("log-stream");

    // Dynamic Metric Elements
    const maeHVAC = document.getElementById("mae-hvac");
    const maeLighting = document.getElementById("mae-lighting");
    const maeMELS = document.getElementById("mae-mels");
    const reconError = document.getElementById("recon-error");

    // Initialize Chart.js
    const ctx = document.getElementById('energyChart').getContext('2d');
    
    Chart.defaults.color = '#94A3B8';
    Chart.defaults.font.family = "'Inter', sans-serif";

    const MAX_DATA_POINTS = 30;

    const chartConfig = {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Actual HVAC',
                    data: [],
                    borderColor: 'rgba(59, 130, 246, 0.4)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Pred HVAC',
                    data: [],
                    borderColor: '#3B82F6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 2,
                    pointHoverRadius: 5
                },
                {
                    label: 'Actual Lighting',
                    data: [],
                    borderColor: 'rgba(20, 184, 166, 0.4)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Pred Lighting',
                    data: [],
                    borderColor: '#14B8A6',
                    tension: 0.4,
                    pointRadius: 2
                },
                {
                    label: 'Actual MELS',
                    data: [],
                    borderColor: 'rgba(139, 92, 246, 0.4)',
                    borderDash: [5, 5],
                    tension: 0.4,
                    pointRadius: 0
                },
                {
                    label: 'Pred MELS',
                    data: [],
                    borderColor: '#8B5CF6',
                    tension: 0.4,
                    pointRadius: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            plugins: {
                tooltip: {
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#fff',
                    bodyColor: '#cbd5e1',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    padding: 12,
                    boxPadding: 6
                },
                legend: {
                    position: 'top',
                    labels: { boxWidth: 12, usePointStyle: true }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    ticks: { maxTicksLimit: 10 }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)', drawBorder: false },
                    beginAtZero: false
                }
            }
        }
    };

    const energyChart = new Chart(ctx, chartConfig);

    // WebSocket Connection Configurations
    let ws;
    
    // IMPORTANT: When hosting the frontend on Vercel, change this variable to your live backend URL!
    // Example: const PROD_WS_URL = "wss://my-nexus-backend.onrender.com/ws/stream";
    const PROD_WS_URL = "wss://nexus-energy-ai.onrender.com/ws/stream"; 

    function connect() {
        let wsUrl;
        
        if (PROD_WS_URL) {
            wsUrl = PROD_WS_URL;
        } else {
            // Automatically connect to the same host/port serving the HTML (Local dev fallback)
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            wsUrl = `${protocol}//${window.location.host}/ws/stream`;
        }
        
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            dot.className = "dot connected";
            text.textContent = "Live Stream Connected";
            addLog("SYSTEM", "WebSocket Connection Established.");
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.error) {
                addLog("ERROR", data.error);
                return;
            }

            processData(data);
        };

        ws.onclose = () => {
            dot.className = "dot disconnected";
            text.textContent = "Disconnected - Reconnecting...";
            addLog("SYSTEM", "WebSocket Disconnected. Reconnecting in 3s...");
            setTimeout(connect, 3000);
        };
        
        ws.onerror = (err) => {
            console.error('Socket encountered error: ', err.message, 'Closing socket');
            ws.close();
        };
    }

    function processData(data) {
        const { timestamp, actual, predicted, metrics } = data;
        
        // Format time 
        const date = new Date(timestamp);
        let timeLabel = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        // if dataset timestamp isn't valid parsing, just use raw string or fallback
        if (timeLabel === "Invalid Date") {
            timeLabel = timestamp.split(' ')[1] || timestamp; 
        }

        // Update Chart
        const chartData = energyChart.data;
        chartData.labels.push(timeLabel);
        
        // Actuals
        chartData.datasets[0].data.push(actual.hvac);
        chartData.datasets[2].data.push(actual.lighting);
        chartData.datasets[4].data.push(actual.mels);
        
        // Preds
        if (predicted) {
            chartData.datasets[1].data.push(predicted.hvac);
            chartData.datasets[3].data.push(predicted.lighting);
            chartData.datasets[5].data.push(predicted.mels);
            
            // Update Metric Cards
            valHVAC.innerText = predicted.hvac.toFixed(2);
            valLighting.innerText = predicted.lighting.toFixed(2);
            valMELS.innerText = predicted.mels.toFixed(2);

            if (metrics) {
                maeHVAC.innerText = metrics.mae.hvac.toFixed(2);
                maeLighting.innerText = metrics.mae.lighting.toFixed(2);
                maeMELS.innerText = metrics.mae.mels.toFixed(2);
                
                reconError.innerText = metrics.reconstruction_error.toFixed(2);
                
                // Visual Alert for high anomaly
                if (metrics.reconstruction_error > 1.5) {
                    reconError.style.color = "#ef4444";
                    reconError.style.textShadow = "0 0 10px rgba(239, 68, 68, 0.8)";
                } else {
                    reconError.style.color = "var(--accent-blue)";
                    reconError.style.textShadow = "none";
                }
            }
            
            // Animate cards briefly
            animateValueChange(valHVAC);
            animateValueChange(valLighting);
            animateValueChange(valMELS);
            
            addLog(timeLabel, `ACTUAL: [H:${actual.hvac.toFixed(1)} L:${actual.lighting.toFixed(1)} M:${actual.mels.toFixed(1)}] | PRED: [H:${predicted.hvac.toFixed(1)} L:${predicted.lighting.toFixed(1)} M:${predicted.mels.toFixed(1)}]`);
        } else {
            // Null preds if buffering
            chartData.datasets[1].data.push(null);
            chartData.datasets[3].data.push(null);
            chartData.datasets[5].data.push(null);
            
            addLog(timeLabel, `Buffering sequence...  [H:${actual.hvac.toFixed(1)} L:${actual.lighting.toFixed(1)} M:${actual.mels.toFixed(1)}]`);
        }

        // Shift old data if exceeding max points
        if (chartData.labels.length > MAX_DATA_POINTS) {
            chartData.labels.shift();
            chartData.datasets.forEach(dataset => dataset.data.shift());
        }

        energyChart.update('none'); // Update without full animation for performance
    }

    function addLog(time, message) {
        const entry = document.createElement("div");
        entry.className = "log-entry";
        entry.innerHTML = `<span class="log-time">${time}</span><span class="log-data">${message}</span>`;
        logStream.appendChild(entry);
        
        // Auto scroll to bottom
        if (logStream.children.length > 50) {
            logStream.removeChild(logStream.firstChild);
        }
        logStream.scrollTop = logStream.scrollHeight;
    }

    function animateValueChange(element) {
        element.style.transform = "scale(1.1)";
        element.style.textShadow = "0 0 15px rgba(255,255,255,0.6)";
        setTimeout(() => {
            element.style.transform = "scale(1)";
            element.style.textShadow = "none";
        }, 150);
    }

    // Start Connection
    connect();
});
