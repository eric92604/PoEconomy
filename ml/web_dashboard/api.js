/**
 * Cloudflare Worker API for PoEconomy Investment Dashboard
 * 
 * This worker serves as the API backend for the live investment dashboard.
 * It fetches data from your PostgreSQL database and serves it to the frontend.
 */

// CORS headers for API responses
const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

// Database configuration - Replace with your actual database URL
const DATABASE_URL = 'YOUR_POSTGRES_CONNECTION_STRING';

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    
    // Handle CORS preflight requests
    if (request.method === 'OPTIONS') {
      return new Response(null, { 
        status: 200, 
        headers: corsHeaders 
      });
    }
    
    try {
      // Route handling
      if (url.pathname === '/api/investment-data') {
        return await handleInvestmentData(request, env);
      }
      
      if (url.pathname === '/api/market-trends') {
        return await handleMarketTrends(request, env);
      }
      
      if (url.pathname === '/api/health') {
        return jsonResponse({ status: 'healthy', timestamp: new Date().toISOString() });
      }
      
      // Serve the main dashboard if no API route matches
      if (url.pathname === '/' || url.pathname === '/index.html') {
        return await serveHTML();
      }
      
      return jsonResponse({ error: 'Not found' }, 404);
      
    } catch (error) {
      console.error('Worker error:', error);
      return jsonResponse({ 
        error: 'Internal server error',
        message: error.message 
      }, 500);
    }
  }
};

/**
 * Handle investment data API endpoint
 */
async function handleInvestmentData(request, env) {
  try {
    // In production, you'd connect to your PostgreSQL database
    // For now, we'll simulate data or fetch from your existing endpoints
    
    const investmentData = await fetchInvestmentData(env);
    
    // Calculate metrics
    const metrics = calculateMetrics(investmentData);
    
    // Get top opportunities
    const opportunities = investmentData
      .filter(item => item.confidence > 0.5)
      .sort((a, b) => {
        const returnA = (a.predicted_price - a.current_price) / a.current_price;
        const returnB = (b.predicted_price - b.current_price) / b.current_price;
        return returnB - returnA;
      })
      .slice(0, 50);
    
    // Generate sample trends data
    const trends = generateTrendsData();
    
    return jsonResponse({
      metrics,
      opportunities,
      trends,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Error fetching investment data:', error);
    return jsonResponse({ error: 'Failed to fetch investment data' }, 500);
  }
}

/**
 * Fetch investment data from your backend
 * In production, this would connect to your PostgreSQL database
 */
async function fetchInvestmentData(env) {
  // Option 1: Direct database connection (if available in Cloudflare Workers)
  // Option 2: Fetch from your existing API endpoints
  // Option 3: Use simulated data for development
  
  // For development, return simulated data
  // Replace this with actual database queries in production
  return [
    {
      currency: "Valdo's Puzzle Box",
      current_price: 48.79,
      predicted_price: 119.51,
      confidence: 0.57,
      volume: 7978,
      trend: 'bullish'
    },
    {
      currency: "Maven's Chisel of Scarabs",
      current_price: 11.00,
      predicted_price: 18.59,
      confidence: 0.79,
      volume: 1833,
      trend: 'bullish'
    },
    {
      currency: "Sacred Blossom",
      current_price: 21.11,
      predicted_price: 48.34,
      confidence: 0.78,
      volume: 567,
      trend: 'bullish'
    },
    {
      currency: "Otherworldly Scouting Report",
      current_price: 3.19,
      predicted_price: 4.42,
      confidence: 0.90,
      volume: 852,
      trend: 'bullish'
    },
    {
      currency: "Mortal Rage",
      current_price: 5.50,
      predicted_price: 5.97,
      confidence: 0.90,
      volume: 151,
      trend: 'bullish'
    },
    // Add more currencies as needed
  ];
}

/**
 * Calculate dashboard metrics from investment data
 */
function calculateMetrics(data) {
  const profitableOpportunities = data.filter(item => {
    const returnPercent = (item.predicted_price - item.current_price) / item.current_price;
    return returnPercent > 0.1 && item.confidence > 0.6; // 10% return with 60% confidence
  });
  
  const bestReturn = Math.max(...data.map(item => 
    (item.predicted_price - item.current_price) / item.current_price * 100
  ));
  
  const avgConfidence = data.reduce((sum, item) => sum + item.confidence, 0) / data.length;
  
  return {
    totalOpportunities: profitableOpportunities.length,
    bestReturn: bestReturn,
    avgConfidence: avgConfidence,
    totalCurrencies: data.length
  };
}

/**
 * Generate sample market trends data
 * In production, this would fetch historical market data
 */
function generateTrendsData() {
  const trends = [];
  const now = new Date();
  
  for (let i = 6; i >= 0; i--) {
    const date = new Date(now);
    date.setDate(date.getDate() - i);
    
    trends.push({
      date: date.toISOString(),
      value: 100 + Math.sin(i * 0.5) * 10 + Math.random() * 5
    });
  }
  
  return trends;
}

/**
 * Handle market trends API endpoint
 */
async function handleMarketTrends(request, env) {
  try {
    const trends = generateTrendsData();
    return jsonResponse({ trends });
  } catch (error) {
    return jsonResponse({ error: 'Failed to fetch market trends' }, 500);
  }
}

/**
 * Serve the main HTML dashboard
 */
async function serveHTML() {
  // In production, you'd fetch the HTML from your storage or embed it
  // For now, we'll redirect to the main dashboard
  const html = `
    <!DOCTYPE html>
    <html>
    <head>
        <title>PoEconomy Dashboard</title>
        <meta http-equiv="refresh" content="0; url=https://your-domain.com/dashboard">
    </head>
    <body>
        <p>Redirecting to dashboard...</p>
    </body>
    </html>
  `;
  
  return new Response(html, {
    headers: {
      'Content-Type': 'text/html',
      ...corsHeaders
    }
  });
}

/**
 * Helper function to create JSON responses
 */
function jsonResponse(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...corsHeaders
    }
  });
}

/**
 * Production database query function
 * Uncomment and modify when you have database access in Workers
 */
/*
async function queryDatabase(query, params = []) {
  try {
    // Use a database adapter like PostgreSQL for Cloudflare Workers
    // or connect to your existing API endpoint
    const response = await fetch('YOUR_API_ENDPOINT', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, params })
    });
    
    return await response.json();
  } catch (error) {
    console.error('Database query error:', error);
    throw error;
  }
}
*/ 