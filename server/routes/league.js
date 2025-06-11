const express = require('express');
const router = express.Router();
const { League } = require('../models');

// GET /api/league - Get all leagues
router.get('/', async (req, res) => {
  try {
    const leagues = await League.findAll({
      order: [['createdAt', 'DESC']]
    });
    res.json(leagues);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/league/active - Get active league
router.get('/active', async (req, res) => {
  try {
    const activeLeague = await League.findOne({
      where: { isActive: true }
    });
    
    if (!activeLeague) {
      return res.status(404).json({ error: 'No active league found' });
    }
    
    res.json(activeLeague);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/league/:id - Get league by ID
router.get('/:id', async (req, res) => {
  try {
    const league = await League.findByPk(req.params.id);
    
    if (!league) {
      return res.status(404).json({ error: 'League not found' });
    }
    
    res.json(league);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router; 