# 🚀 PoEconomy Live Investment Dashboard - Deployment Guide

This guide will walk you through deploying your live, interactive investment dashboard to your Cloudflare domain.

## 📋 Prerequisites

- Cloudflare account with your domain configured
- Access to your PostgreSQL database
- Your ML models and Python scripts working locally

## 🏗️ Deployment Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Cloudflare     │    │  Cloudflare      │    │  Your Database  │
│  Pages          │◄───│  Workers         │◄───│  (PostgreSQL)   │
│  (Frontend)     │    │  (API Backend)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🎯 Step 1: Deploy the Frontend (Cloudflare Pages)

### 1.1 Prepare Your Files
```bash
# Create a deployment folder
mkdir dashboard-deploy
cd dashboard-deploy

# Copy the dashboard files
cp ../server/public/index.html ./
```

### 1.2 Deploy to Cloudflare Pages
1. Go to **Cloudflare Dashboard** → **Pages**
2. Click **"Create a project"** → **"Upload assets"**
3. Name your project: `poeconomy-dashboard`
4. Upload your `index.html` file
5. Click **"Save and Deploy"**

### 1.3 Configure Custom Domain
1. In your Pages project, go to **"Custom domains"**
2. Click **"Set up a custom domain"**
3. Enter your domain (e.g., `dashboard.yourdomain.com`)
4. Cloudflare will automatically configure DNS

## ⚡ Step 2: Deploy the API Backend (Cloudflare Workers)

### 2.1 Install Wrangler CLI
```bash
npm install -g wrangler
```

### 2.2 Login to Cloudflare
```bash
wrangler login
```

### 2.3 Create Worker Project
```bash
# Create new worker project
wrangler init poeconomy-api
cd poeconomy-api

# Copy the API code
cp ../server/workers/api.js ./src/worker.js
```

### 2.4 Configure wrangler.toml
```toml
name = "poeconomy-api"
main = "src/worker.js"
compatibility_date = "2024-01-01"

[env.production]
name = "poeconomy-api"
route = "api.yourdomain.com/*"

# Add environment variables for database connection
[env.production.vars]
DATABASE_URL = "your-postgres-connection-string"
```

### 2.5 Deploy the Worker
```bash
wrangler publish --env production
```

## 🔄 Step 3: Connect to Your Database

### Option A: Direct Database Connection (Recommended)

1. **Install Database Adapter**
```bash
npm install @cloudflare/workers-types
```

2. **Update your API Worker** to connect to PostgreSQL:
```javascript
// Add to your worker.js
async function connectToDatabase() {
  // Use a database adapter compatible with Cloudflare Workers
  // Example: Neon, PlanetScale, or Supabase
  const response = await fetch('YOUR_DATABASE_API_ENDPOINT', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${env.DATABASE_TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query: 'SELECT * FROM live_currency_prices WHERE league = $1 LIMIT 100',
      params: ['Mercenaries']
    })
  });
  
  return await response.json();
}
```

### Option B: Proxy Through Your Existing Server

1. **Modify your Express server** to allow CORS:
```javascript
// Add to your server/index.js
app.use(cors({
  origin: 'https://dashboard.yourdomain.com'
}));

// Add API endpoint
app.get('/api/investment-data', async (req, res) => {
  try {
    // Use your existing Python script
    const { spawn } = require('child_process');
    const python = spawn('python', ['../ml/scripts/generate_investment_report.py', '--json']);
    
    let data = '';
    python.stdout.on('data', (chunk) => {
      data += chunk;
    });
    
    python.on('close', (code) => {
      if (code === 0) {
        res.json(JSON.parse(data));
      } else {
        res.status(500).json({ error: 'Failed to generate data' });
      }
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});
```

2. **Update your Worker** to proxy requests:
```javascript
async function fetchInvestmentData(env) {
  const response = await fetch('https://your-server.com/api/investment-data');
  return await response.json();
}
```

## 🔧 Step 4: Configure Real-time Updates

### 4.1 Create a Scheduled Worker for Data Updates
```bash
wrangler init poeconomy-scheduler
```

```javascript
// scheduler.js
export default {
  async scheduled(event, env, ctx) {
    // Trigger your Python script to generate new reports
    try {
      const response = await fetch('https://your-server.com/api/generate-report', {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${env.API_KEY}` }
      });
      
      console.log('Report generated:', await response.text());
    } catch (error) {
      console.error('Failed to generate report:', error);
    }
  }
};
```

### 4.2 Schedule the Worker
```toml
# Add to wrangler.toml
[triggers]
crons = ["*/15 * * * *"]  # Run every 15 minutes
```

## 🛡️ Step 5: Security & Performance

### 5.1 Add Authentication (Optional)
```javascript
// Add to worker.js
async function authenticateRequest(request) {
  const apiKey = request.headers.get('X-API-Key');
  if (!apiKey || apiKey !== env.API_KEY) {
    return new Response('Unauthorized', { status: 401 });
  }
  return null;
}
```

### 5.2 Add Caching
```javascript
// Add caching headers
const response = jsonResponse(data);
response.headers.set('Cache-Control', 'public, max-age=300'); // 5 minutes
return response;
```

### 5.3 Add Rate Limiting
```javascript
// Simple rate limiting
const clientIP = request.headers.get('CF-Connecting-IP');
// Implement rate limiting logic
```

## 📊 Step 6: Monitoring & Analytics

### 6.1 Enable Cloudflare Analytics
1. Go to **Analytics & Logs** in Cloudflare Dashboard
2. Enable **Web Analytics** for your Pages domain
3. Monitor Worker performance in **Workers** → **Analytics**

### 6.2 Add Custom Metrics
```javascript
// Add to your worker
console.log('API Request:', {
  endpoint: url.pathname,
  timestamp: new Date().toISOString(),
  userAgent: request.headers.get('User-Agent')
});
```

## 🚀 Step 7: Go Live!

1. **Test your deployment:**
   - Visit `https://dashboard.yourdomain.com`
   - Check that data loads properly
   - Verify real-time updates work

2. **Monitor performance:**
   - Check Cloudflare Analytics
   - Monitor Worker logs
   - Verify database connections

3. **Set up alerts:**
   - Configure Cloudflare notifications
   - Set up uptime monitoring

## 🔧 Troubleshooting

### Common Issues:

1. **CORS Errors:**
   - Ensure your Worker has proper CORS headers
   - Check that your domain is whitelisted

2. **Database Connection Issues:**
   - Verify connection string is correct
   - Check firewall settings
   - Ensure database allows connections from Cloudflare IPs

3. **Worker Timeout:**
   - Optimize database queries
   - Add proper error handling
   - Consider using Durable Objects for persistent connections

4. **Data Not Updating:**
   - Check scheduled Worker logs
   - Verify Python script execution
   - Monitor database write operations

## 📝 Next Steps

1. **Custom Domain Setup:** Configure `dashboard.yourdomain.com`
2. **SSL/TLS:** Ensure HTTPS is properly configured
3. **Performance Optimization:** Add caching layers
4. **User Authentication:** Add login functionality if needed
5. **Mobile Optimization:** Test and optimize for mobile devices

## 💡 Tips for Success

- **Start Simple:** Deploy basic functionality first, then add features
- **Monitor Closely:** Watch for errors and performance issues
- **Cache Smartly:** Use appropriate cache headers for different data types
- **Plan for Scale:** Consider rate limits and resource usage
- **Backup Strategy:** Ensure your data and configurations are backed up

Your live investment dashboard will now be accessible at your custom domain with real-time updates, interactive charts, and professional presentation! 🎉 