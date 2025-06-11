const express = require('express');
const router = express.Router();
const { Item, ItemPrice, League } = require('../models');
const { Op } = require('sequelize');

// GET /api/item - Get all items
router.get('/', async (req, res) => {
  try {
    const { limit = 100, offset = 0, search, category, type } = req.query;
    
    const whereClause = {};
    if (search) {
      whereClause.name = {
        [Op.iLike]: `%${search}%`
      };
    }
    if (category) {
      whereClause.category = category;
    }
    if (type) {
      whereClause.type = type;
    }
    
    const items = await Item.findAll({
      where: whereClause,
      order: [['name', 'ASC']],
      limit: parseInt(limit),
      offset: parseInt(offset)
    });
    
    res.json(items);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/item/prices/:leagueId - Get item prices for a league
router.get('/prices/:leagueId', async (req, res) => {
  try {
    const { leagueId } = req.params;
    const { limit = 100, offset = 0, date, minValue, maxValue } = req.query;
    
    const whereClause = { leagueId };
    if (date) {
      whereClause.date = date;
    }
    if (minValue) {
      whereClause.value = { [Op.gte]: parseFloat(minValue) };
    }
    if (maxValue) {
      whereClause.value = { ...whereClause.value, [Op.lte]: parseFloat(maxValue) };
    }
    
    const prices = await ItemPrice.findAll({
      where: whereClause,
      include: [
        { model: Item, as: 'item' },
        { model: League, as: 'league' }
      ],
      order: [['date', 'DESC'], ['value', 'DESC']],
      limit: parseInt(limit),
      offset: parseInt(offset)
    });
    
    res.json(prices);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/item/history/:itemId - Get price history for an item
router.get('/history/:itemId', async (req, res) => {
  try {
    const { itemId } = req.params;
    const { leagueId, days = 30, variant, links } = req.query;
    
    const whereClause = { itemId };
    
    if (leagueId) {
      whereClause.leagueId = leagueId;
    }
    
    if (variant) {
      whereClause.variant = variant;
    }
    
    if (links) {
      whereClause.links = parseInt(links);
    }
    
    if (days) {
      const daysAgo = new Date();
      daysAgo.setDate(daysAgo.getDate() - parseInt(days));
      whereClause.date = {
        [Op.gte]: daysAgo
      };
    }
    
    const history = await ItemPrice.findAll({
      where: whereClause,
      include: [
        { model: Item, as: 'item' },
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