const express = require('express');
const router = express.Router();
const { Currency, CurrencyPrice, League } = require('../models');
const { Op } = require('sequelize');

// GET /api/currency - Get all currencies
router.get('/', async (req, res) => {
  try {
    const currencies = await Currency.findAll({
      order: [['name', 'ASC']]
    });
    res.json(currencies);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/currency/prices/:leagueId - Get currency prices for a league
router.get('/prices/:leagueId', async (req, res) => {
  try {
    const { leagueId } = req.params;
    const { limit = 100, offset = 0, date } = req.query;
    
    const whereClause = { leagueId };
    if (date) {
      whereClause.date = date;
    }
    
    const prices = await CurrencyPrice.findAll({
      where: whereClause,
      include: [
        { model: Currency, as: 'getCurrency' },
        { model: Currency, as: 'payCurrency' },
        { model: League, as: 'league' }
      ],
      order: [['date', 'DESC']],
      limit: parseInt(limit),
      offset: parseInt(offset)
    });
    
    res.json(prices);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/currency/history/:getCurrencyId/:payCurrencyId - Get price history
router.get('/history/:getCurrencyId/:payCurrencyId', async (req, res) => {
  try {
    const { getCurrencyId, payCurrencyId } = req.params;
    const { leagueId, days = 30 } = req.query;
    
    const whereClause = {
      getCurrencyId,
      payCurrencyId
    };
    
    if (leagueId) {
      whereClause.leagueId = leagueId;
    }
    
    if (days) {
      const daysAgo = new Date();
      daysAgo.setDate(daysAgo.getDate() - parseInt(days));
      whereClause.date = {
        [Op.gte]: daysAgo
      };
    }
    
    const history = await CurrencyPrice.findAll({
      where: whereClause,
      include: [
        { model: Currency, as: 'getCurrency' },
        { model: Currency, as: 'payCurrency' },
        { model: League, as: 'league' }
      ],
      order: [['date', 'ASC']]
    });
    
    res.json(history);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router; 