const express = require('express');
const { spawn } = require('child_process');
const path = require('path');
const router = express.Router();

// Cache for dashboard data to reduce database load
let dashboardCache = {
  data: null,
  timestamp: null,
  ttl: 5 * 60 * 1000 // 5 minutes TTL
};

/**
 * GET /api/dashboard/investment-data - Get current investment opportunities
 */
router.get('/investment-data', async (req, res) => {
  try {
    // Check cache first
    const now = Date.now();
    if (dashboardCache.data && 
        dashboardCache.timestamp && 
        (now - dashboardCache.timestamp) < dashboardCache.ttl) {
      return res.json(dashboardCache.data);
    }

    // Generate fresh data using Python script
    const dashboardData = await generateDashboardData();
    
    // Update cache
    dashboardCache.data = dashboardData;
    dashboardCache.timestamp = now;
    
    res.json(dashboardData);
    
  } catch (error) {
    console.error('Error fetching investment data:', error);
    res.status(500).json({ 
      error: 'Failed to fetch investment data',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * GET /api/dashboard/market-trends - Get market trends data
 */
router.get('/market-trends', async (req, res) => {
  try {
    const days = parseInt(req.query.days) || 7;
    
    // For now, this is included in the main dashboard data
    // In the future, this could be a separate endpoint
    const dashboardData = await generateDashboardData();
    
    res.json({
      trends: dashboardData.trends || [],
      timestamp: dashboardData.timestamp
    });
    
  } catch (error) {
    console.error('Error fetching market trends:', error);
    res.status(500).json({ 
      error: 'Failed to fetch market trends',
      message: error.message 
    });
  }
});

/**
 * GET /api/dashboard/metrics - Get dashboard metrics only
 */
router.get('/metrics', async (req, res) => {
  try {
    const dashboardData = await generateDashboardData();
    
    res.json({
      metrics: dashboardData.metrics || {},
      timestamp: dashboardData.timestamp,
      status: dashboardData.status
    });
    
  } catch (error) {
    console.error('Error fetching metrics:', error);
    res.status(500).json({ 
      error: 'Failed to fetch metrics',
      message: error.message 
    });
  }
});

/**
 * POST /api/dashboard/refresh - Force refresh dashboard data
 */
router.post('/refresh', async (req, res) => {
  try {
    // Clear cache to force fresh data
    dashboardCache.data = null;
    dashboardCache.timestamp = null;
    
    const dashboardData = await generateDashboardData();
    
    // Update cache
    dashboardCache.data = dashboardData;
    dashboardCache.timestamp = Date.now();
    
    res.json({
      message: 'Dashboard data refreshed successfully',
      data: dashboardData,
      refreshed_at: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Error refreshing dashboard data:', error);
    res.status(500).json({ 
      error: 'Failed to refresh dashboard data',
      message: error.message 
    });
  }
});

/**
 * GET /api/dashboard/health - Health check endpoint
 */
router.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    service: 'PoEconomy Dashboard API',
    timestamp: new Date().toISOString(),
    cache_status: {
      has_data: !!dashboardCache.data,
      last_update: dashboardCache.timestamp ? new Date(dashboardCache.timestamp).toISOString() : null,
      ttl_remaining: dashboardCache.timestamp ? 
        Math.max(0, dashboardCache.ttl - (Date.now() - dashboardCache.timestamp)) : 0
    }
  });
});

/**
 * GET /api/dashboard/status - Detailed status information
 */
router.get('/status', async (req, res) => {
  try {
    // Check if Python script is available
    const pythonScriptPath = path.join(__dirname, '../../ml/scripts/generate_dashboard_data.py');
    const fs = require('fs');
    const scriptExists = fs.existsSync(pythonScriptPath);
    
    res.json({
      service: 'PoEconomy Dashboard API',
      timestamp: new Date().toISOString(),
      python_script: {
        path: pythonScriptPath,
        exists: scriptExists
      },
      cache: {
        has_data: !!dashboardCache.data,
        last_update: dashboardCache.timestamp ? new Date(dashboardCache.timestamp).toISOString() : null,
        ttl_seconds: dashboardCache.ttl / 1000,
        size_bytes: dashboardCache.data ? JSON.stringify(dashboardCache.data).length : 0
      },
      system: {
        node_version: process.version,
        platform: process.platform,
        memory_usage: process.memoryUsage()
      }
    });
    
  } catch (error) {
    res.status(500).json({
      error: 'Failed to get status',
      message: error.message
    });
  }
});

/**
 * Generate dashboard data using Python script
 */
async function generateDashboardData() {
  return new Promise((resolve, reject) => {
    try {
      const pythonScriptPath = path.join(__dirname, '../../ml/scripts/generate_dashboard_data.py');
      
      // Spawn Python process
      const python = spawn('python', [pythonScriptPath, '--stdout'], {
        cwd: path.join(__dirname, '../../ml/scripts'),
        stdio: ['pipe', 'pipe', 'pipe']
      });
      
      let stdout = '';
      let stderr = '';
      
      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });
      
      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });
      
      python.on('close', (code) => {
        if (code === 0) {
          try {
            const dashboardData = JSON.parse(stdout);
            resolve(dashboardData);
          } catch (parseError) {
            console.error('Failed to parse Python script output:', parseError);
            console.error('Raw output:', stdout);
            reject(new Error(`Failed to parse dashboard data: ${parseError.message}`));
          }
        } else {
          console.error('Python script failed with code:', code);
          console.error('Error output:', stderr);
          
          // Return sample data as fallback
          resolve(generateFallbackData());
        }
      });
      
      python.on('error', (error) => {
        console.error('Failed to spawn Python process:', error);
        // Return sample data as fallback
        resolve(generateFallbackData());
      });
      
      // Set timeout to prevent hanging
      setTimeout(() => {
        python.kill('SIGTERM');
        reject(new Error('Python script execution timeout'));
      }, 30000); // 30 second timeout
      
    } catch (error) {
      console.error('Error in generateDashboardData:', error);
      resolve(generateFallbackData());
    }
  });
}

/**
 * Generate fallback data when Python script fails
 */
function generateFallbackData() {
  const sampleCurrencies = [
    "Valdo's Puzzle Box", "Maven's Chisel of Scarabs", "Sacred Blossom",
    "Otherworldly Scouting Report", "Mortal Rage", "Dedication to the Goddess"
  ];
  
  const opportunities = sampleCurrencies.map(currency => {
    const currentPrice = Math.random() * 50 + 5;
    const returnPercent = (Math.random() - 0.3) * 100;
    const predictedPrice = currentPrice * (1 + returnPercent / 100);
    const confidence = Math.random() * 0.4 + 0.6;
    
    return {
      currency,
      current_price: Math.round(currentPrice * 100) / 100,
      predicted_price: Math.round(predictedPrice * 100) / 100,
      return_percent: Math.round(returnPercent * 10) / 10,
      confidence: Math.round(confidence * 100) / 100,
      risk_level: confidence > 0.8 ? 'LOW' : confidence > 0.6 ? 'MEDIUM' : 'HIGH',
      recommendation: returnPercent > 20 ? 'BUY' : returnPercent < -10 ? 'SELL' : 'HOLD',
      volume: Math.floor(Math.random() * 1000) + 100,
      model_type: 'fallback',
      timestamp: new Date().toISOString()
    };
  });
  
  // Generate trends data
  const trends = [];
  for (let i = 6; i >= 0; i--) {
    const date = new Date();
    date.setDate(date.getDate() - i);
    trends.push({
      date: date.toISOString().split('T')[0],
      value: 100 + Math.sin(i * 0.5) * 10 + (Math.random() - 0.5) * 5,
      currency_count: 50,
      total_change: (Math.random() - 0.5) * 10
    });
  }
  
  return {
    timestamp: new Date().toISOString(),
    metrics: {
      total_opportunities: opportunities.filter(o => o.return_percent > 10).length,
      best_return: Math.max(...opportunities.map(o => o.return_percent)),
      avg_confidence: opportunities.reduce((sum, o) => sum + o.confidence, 0) / opportunities.length,
      total_currencies: opportunities.length,
      profitable_count: opportunities.filter(o => o.return_percent > 0).length,
      high_confidence_count: opportunities.filter(o => o.confidence > 0.7).length
    },
    opportunities,
    trends,
    status: 'fallback_data',
    data_freshness: {
      predictions_generated: opportunities.length,
      trends_days: trends.length,
      last_update: new Date().toISOString()
    }
  };
}

module.exports = router; 