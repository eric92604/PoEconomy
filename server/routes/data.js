const express = require('express');
const router = express.Router();
const { importAllHistoricalData } = require('../scripts/import-historical-data');

// POST /api/data/import-historical - Import historical data
router.post('/import-historical', async (req, res) => {
  try {
    console.log('🔄 Starting historical data import via API...');
    await importAllHistoricalData();
    res.json({ 
      success: true, 
      message: 'Historical data import completed successfully' 
    });
  } catch (error) {
    console.error('❌ Historical data import failed:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message 
    });
  }
});

// GET /api/data/status - Get data status
router.get('/status', async (req, res) => {
  try {
    const { sequelize, League, Currency, Item, CurrencyPrice, ItemPrice } = require('../models');
    
    const stats = {
      leagues: await League.count(),
      currencies: await Currency.count(),
      items: await Item.count(),
      currencyPrices: await CurrencyPrice.count(),
      itemPrices: await ItemPrice.count(),
      timestamp: new Date().toISOString()
    };
    
    res.json(stats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router; 