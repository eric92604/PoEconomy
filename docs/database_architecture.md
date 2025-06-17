# Database Architecture: PoEconomy

## Overview

This document describes the current PostgreSQL database schema for the PoEconomy project, including all tables, relationships, and indexes. The database has evolved significantly to support live data ingestion, machine learning predictions, and real-time market analysis.

## Entity-Relationship Summary

- **currency** (1) ───< (N) **currency_prices** (via getCurrencyId)
- **currency** (1) ───< (N) **currency_prices** (via payCurrencyId)
- **items** (1) ───< (N) **item_prices** (via itemId)
- **leagues** (1) ───< (N) **currency_prices** (via leagueId)
- **leagues** (1) ───< (N) **item_prices** (via leagueId)

---

## Core Tables

### Table: currency

**Row Count**: 236

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('currencies_id_seq'::regclass) | PK |
| name | character varying(255) | no |  | Unique currency name |
| isBaseCurrency | boolean | yes | false | Whether this is a base currency |
| created | timestamp with time zone | no | now() | Record creation time |
| updated | timestamp with time zone | no | now() | Last update time |
| isAvailableInCurrentLeague | boolean | yes | true | Current league availability |
| lastAvailabilityCheck | timestamp with time zone | yes | now() | Last availability check |
| availabilitySource | character varying(50) | yes | 'manual' | Source of availability data |

**Indexes:**
- PRIMARY KEY: currency_pkey
- UNIQUE: currency_name_key
- INDEX: currencies_name
- INDEX: currencies_is_base_currency
- INDEX: idx_currency_availability

---

### Table: leagues

**Row Count**: 47

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('leagues_id_seq'::regclass) | PK |
| name | character varying | no |  | Unique league name |
| startDate | timestamp with time zone | yes |  | League start date |
| endDate | timestamp with time zone | yes |  | League end date |
| isActive | boolean | yes | false | Whether league is currently active |
| version | character varying | yes |  | Game version |
| created | timestamp with time zone | yes | now() | Record creation time |
| updated | timestamp with time zone | yes | now() | Last update time |

**Indexes:**
- PRIMARY KEY: leagues_pkey
- UNIQUE: leagues_name_key
- INDEX: leagues_name
- INDEX: leagues_is_active

---

### Table: currency_prices

**Row Count**: 367,174

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('currency_prices_id_seq'::regclass) | PK |
| leagueId | integer | no |  | FK → leagues(id) |
| getCurrencyId | integer | no |  | FK → currency(id) |
| payCurrencyId | integer | no |  | FK → currency(id) |
| date | date | no |  | Price date |
| value | numeric | no |  | Exchange rate value |
| confidence | USER-DEFINED | no |  | Confidence level enum |
| created | timestamp with time zone | no | now() | Record creation time |
| updated | timestamp with time zone | no | now() | Last update time |

**Indexes:**
- PRIMARY KEY: currency_prices_pkey
- UNIQUE: currency_prices_league_id_get_currency_id_pay_currency_id_date
- INDEX: currency_prices_date
- INDEX: currency_prices_confidence
- INDEX: currency_prices_get_currency_id_pay_currency_id
- INDEX: currency_prices_league_id_date

**Foreign Keys:**
- leagueId → leagues.id
- getCurrencyId → currency.id
- payCurrencyId → currency.id

---

### Table: items

**Row Count**: 0

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('items_id_seq'::regclass) | PK |
| name | character varying(255) | no |  | Item name |
| type | character varying(255) | yes |  | Item type |
| baseType | character varying(255) | yes |  | Base item type |
| variant | character varying(255) | yes |  | Item variant |
| links | character varying(255) | yes |  | Socket links |
| created | timestamp with time zone | no | now() | Record creation time |
| updated | timestamp with time zone | no | now() | Last update time |

**Indexes:**
- PRIMARY KEY: items_pkey
- INDEX: items_name
- INDEX: items_type
- INDEX: items_base_type
- INDEX: items_category

---

### Table: item_prices

**Row Count**: 0

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('item_prices_id_seq'::regclass) | PK |
| leagueId | integer | no |  | FK → leagues(id) |
| itemId | integer | no |  | FK → items(id) |
| date | date | no |  | Price date |
| value | numeric | no |  | Price in Chaos Orbs |
| confidence | USER-DEFINED | no |  | Confidence level enum |
| variant | character varying(255) | yes |  | Item variant |
| links | integer | yes |  | Number of linked sockets |
| volume | integer | yes |  | Trading volume |
| source | character varying(255) | yes |  | Data source |
| corrupted | boolean | yes |  | Whether item is corrupted |
| quality | integer | yes |  | Item quality percentage |
| ilvl | integer | yes |  | Item level |
| created | timestamp with time zone | no | now() | Record creation time |
| updated | timestamp with time zone | no | now() | Last update time |

**Indexes:**
- PRIMARY KEY: item_prices_pkey
- UNIQUE: item_prices_league_id_item_id_date_variant_links
- INDEX: item_prices_date
- INDEX: item_prices_confidence
- INDEX: item_prices_item_id
- INDEX: item_prices_item_id_date
- INDEX: item_prices_league_id
- INDEX: item_prices_league_id_date
- INDEX: item_prices_league_id_item_id_date
- INDEX: item_prices_links
- INDEX: item_prices_variant

**Foreign Keys:**
- leagueId → leagues.id
- itemId → items.id

---

## Live Data Tables

### Table: live_currency_prices

**Row Count**: 34,968

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('live_currency_prices_id_seq'::regclass) | PK |
| currency_name | character varying(100) | no |  | Currency name |
| league | character varying(50) | no |  | League name |
| direction | character varying(10) | no |  | Trade direction (buy/sell) |
| value | numeric | no |  | Exchange rate |
| count | integer | yes |  | Number of listings |
| chaos_equivalent | numeric | yes |  | Value in Chaos Orbs |
| listing_count | integer | yes |  | Total listings |
| confidence_level | character varying(20) | yes |  | Confidence level |
| total_change | numeric | yes |  | Price change percentage |
| sparkline | text | yes |  | Price history sparkline |
| sample_time | timestamp with time zone | no |  | Data sample timestamp |
| created_at | timestamp with time zone | yes | now() | Record creation time |
| data_source | character varying(50) | yes | 'poe_ninja' | Data source identifier |
| api_version | character varying(10) | yes | '1.0' | API version |

**Indexes:**
- PRIMARY KEY: live_currency_prices_pkey
- UNIQUE: live_currency_prices_currency_name_league_direction_sample_time_key
- INDEX: idx_live_currency_prices_currency_league
- INDEX: idx_live_currency_prices_sample_time

---

### Table: live_poe_watch

**Row Count**: 10,951

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('live_poe_watch_id_seq'::regclass) | PK |
| poe_watch_id | integer | no |  | POE Watch API ID |
| currency_name | character varying(100) | no |  | Currency name |
| category | character varying(50) | yes |  | Currency category |
| group_name | character varying(50) | yes |  | Currency group |
| frame | integer | yes |  | Item frame type |
| icon_url | text | yes |  | Currency icon URL |
| mean_price | numeric | yes |  | Mean price |
| min_price | numeric | yes |  | Minimum price |
| max_price | numeric | yes |  | Maximum price |
| exalted_price | numeric | yes |  | Price in Exalted Orbs |
| divine_price | numeric | yes |  | Price in Divine Orbs |
| daily_volume | integer | yes |  | Daily trading volume |
| price_change_percent | numeric | yes |  | Price change percentage |
| price_history | jsonb | yes |  | Historical price data |
| low_confidence | boolean | yes |  | Low confidence indicator |
| league | character varying(50) | no |  | League name |
| mode_price | numeric | yes |  | Mode price |
| total_listings | integer | yes |  | Total listings |
| current_listings | integer | yes |  | Current listings |
| accepted_listings | integer | yes |  | Accepted listings |
| fetch_time | timestamp with time zone | yes | now() | Data fetch timestamp |

**Indexes:**
- PRIMARY KEY: live_poe_watch_pkey
- INDEX: idx_poe_watch_currency_name
- INDEX: idx_poe_watch_id_league
- INDEX: idx_poe_watch_league_time
- INDEX: idx_poe_watch_price_change

---

## Machine Learning Tables

### Table: live_predictions

**Row Count**: 1,165

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('live_predictions_id_seq'::regclass) | PK |
| currency | character varying(100) | no |  | Currency name |
| current_price | numeric | no |  | Current market price |
| predicted_price | numeric | no |  | ML predicted price |
| price_change_percent | numeric | yes |  | Predicted change percentage |
| confidence_score | numeric | yes |  | Model confidence score |
| prediction_horizon_days | integer | yes |  | Prediction time horizon |
| model_type | character varying(50) | yes |  | ML model type used |
| features_used | integer | yes |  | Number of features used |
| data_points_used | integer | yes |  | Training data points |
| prediction_time | timestamp with time zone | yes | now() | Prediction timestamp |

**Indexes:**
- PRIMARY KEY: live_predictions_pkey
- INDEX: idx_live_predictions_currency_time

---

## Monitoring Tables

### Table: price_alerts

**Row Count**: 3,932

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | no | nextval('price_alerts_id_seq'::regclass) | PK |
| currency_name | character varying(100) | no |  | Currency name |
| league | character varying(50) | no |  | League name |
| change_percent | numeric | no |  | Price change percentage |
| chaos_equivalent | numeric | yes |  | Value in Chaos Orbs |
| confidence_level | character varying(20) | yes |  | Confidence level |
| alert_time | timestamp with time zone | yes | now() | Alert timestamp |
| alert_message | text | yes |  | Alert message |

**Indexes:**
- PRIMARY KEY: price_alerts_pkey

---

## Views

### View: vw_currency_prices

**Row Count**: 367,174

A materialized view providing a simplified interface to currency price data with human-readable league and currency names.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| id | integer | yes |  | Original price record ID |
| league | character varying | yes |  | League name |
| getcurrency | character varying(255) | yes |  | Currency being obtained |
| paycurrency | character varying(255) | yes |  | Currency being paid |
| date | date | yes |  | Price date |
| value | numeric | yes |  | Exchange rate |
| confidence | USER-DEFINED | yes |  | Confidence level |
| created | timestamp with time zone | yes |  | Record creation time |
| updated | timestamp with time zone | yes |  | Last update time |

---

## Database Statistics

- **Total Tables**: 10 (7 tables + 1 view + 2 live data tables)
- **Total Records**: 785,647
- **Total Indexes**: 43
- **Total Foreign Keys**: 5

### Table Sizes

- **currency_prices**: 367,174 records (Historical price data)
- **vw_currency_prices**: 367,174 records (Price data view)
- **live_currency_prices**: 34,968 records (Live PoE.ninja data)
- **live_poe_watch**: 10,951 records (Live POE Watch data)
- **price_alerts**: 3,932 records (Price change alerts)
- **live_predictions**: 1,165 records (ML predictions)
- **currency**: 236 records (Currency definitions)
- **leagues**: 47 records (League information)
- **items**: 0 records (Item definitions - unused)
- **item_prices**: 0 records (Item prices - unused)

---

## Key Features

### Live Data Integration
- **Real-time data ingestion** from PoE.ninja and POE Watch APIs
- **Automated price monitoring** with configurable alerts
- **Multi-source data validation** and confidence scoring

### Machine Learning Support
- **Prediction storage** with model metadata and confidence scores
- **Multi-horizon forecasting** (1, 3, 7 days)
- **Model performance tracking** and feature usage statistics

### Data Quality
- **Comprehensive indexing** for optimal query performance
- **Foreign key constraints** ensuring data integrity
- **Timestamp tracking** for all data modifications
- **Confidence levels** for data quality assessment

### Scalability
- **Partitioned by league and date** for efficient querying
- **Optimized indexes** for common query patterns
- **View-based abstractions** for simplified data access

---

*Documentation generated automatically from database schema analysis* 