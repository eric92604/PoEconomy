const express = require('express');
const router = express.Router();
const { Prediction, Currency, Item, League } = require('../models');

// GET /api/prediction/:leagueId - Get predictions for a league
router.get('/:leagueId', async (req, res) => {
  try {
    const { leagueId } = req.params;
    const { type, timeframe, limit = 50 } = req.query;
    
    const whereClause = { leagueId };
    if (type) {
      whereClause.predictionType = type;
    }
    if (timeframe) {
      whereClause.timeframe = timeframe;
    }
    
    const predictions = await Prediction.findAll({
      where: whereClause,
      include: [
        { model: Currency, as: 'currency' },
        { model: Item, as: 'item' },
        { model: League, as: 'league' }
      ],
      order: [['confidence', 'DESC'], ['createdAt', 'DESC']],
      limit: parseInt(limit)
    });
    
    res.json(predictions);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// GET /api/prediction/opportunities/:leagueId - Get investment opportunities
router.get('/opportunities/:leagueId', async (req, res) => {
  try {
    const { leagueId } = req.params;
    const { minConfidence = 0.7, limit = 20 } = req.query;
    
    const opportunities = await Prediction.findAll({
      where: {
        leagueId,
        confidence: { [require('sequelize').Op.gte]: parseFloat(minConfidence) },
        trend: ['bullish']
      },
      include: [
        { model: Currency, as: 'currency' },
        { model: Item, as: 'item' },
        { model: League, as: 'league' }
      ],
      order: [['confidence', 'DESC'], ['predictedPrice', 'DESC']],
      limit: parseInt(limit)
    });
    
    res.json(opportunities);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router; 