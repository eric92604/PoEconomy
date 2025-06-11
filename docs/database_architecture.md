# Database Architecture: PoEconomy

## Entity-Relationship Diagram (Textual)

- **leagues** (1) ───< (N) **currency_prices** >─── (N) **currency** (as get/pay)
- **leagues** (1) ───< (N) **item_prices** >─── (N) **items**
- **currency** (1) ───< (N) **predictions**

---

## Table: leagues
| Column     | Type                      | Nullable | Default                | Description         |
|------------|---------------------------|----------|------------------------|---------------------|
| id         | integer                   | no       | nextval('leagues_id_seq'::regclass) | PK |
| name       | character varying         | no       |                        | Unique, indexed     |
| startDate  | timestamp with time zone  | yes      |                        |                     |
| endDate    | timestamp with time zone  | yes      |                        |                     |
| isActive   | boolean                   | yes      | false                  | Indexed             |
| version    | character varying         | yes      |                        |                     |
| created    | timestamp with time zone  | yes      | now()                  |                     |
| updated    | timestamp with time zone  | yes      | now()                  |                     |

Indexes: PK (id), UNIQUE (name), isActive, name

---

## Table: currency
| Column         | Type                    | Nullable | Default                | Description         |
|----------------|------------------------|----------|------------------------|---------------------|
| id             | integer                 | no       | nextval('currencies_id_seq'::regclass) | PK |
| name           | character varying(255)  | no       |                        | Unique, indexed     |
| isBaseCurrency | boolean                 | yes      | false                  | Indexed             |
| created        | timestamp with time zone| no       |                        |                     |
| updated        | timestamp with time zone| no       |                        |                     |

Indexes: PK (id), UNIQUE (name), isBaseCurrency, name

---

## Table: currency_prices
| Column         | Type                              | Nullable | Default | Description |
|----------------|-----------------------------------|----------|---------|-------------|
| id             | integer                           | no       |         | PK          |
| leagueId       | integer                           | no       |         | FK → leagues(id) |
| getCurrencyId  | integer                           | no       |         | FK → currency(id) |
| payCurrencyId  | integer                           | no       |         | FK → currency(id) |
| date           | date                              | no       |         | Indexed     |
| value          | numeric(10,5)                     | no       |         |              |
| confidence     | enum_currency_prices_confidence   | no       |         | Indexed     |
| created        | timestamp with time zone          | no       | now()   |              |
| updated        | timestamp with time zone          | no       | now()   |              |

Indexes: PK (id), confidence, date, (getCurrencyId, payCurrencyId), (leagueId, date), UNIQUE (leagueId, getCurrencyId, payCurrencyId, date)

---

## Table: item_prices
| Column      | Type                             | Nullable | Default | Description |
|-------------|----------------------------------|----------|---------|-------------|
| id          | integer                          | no       |         | PK          |
| leagueId    | integer                          | no       |         | FK → leagues(id) |
| itemId      | integer                          | no       |         | FK → items(id) |
| date        | date                             | no       |         | Indexed     |
| value       | numeric(10,2)                    | no       |         | Price in Chaos Orbs |
| confidence  | enum_item_prices_confidence      | no       |         | Indexed     |
| variant     | character varying(255)           | yes      |         | Item variant |
| links       | integer                          | yes      |         | Linked sockets |
| volume      | integer                          | yes      |         |              |
| source      | character varying(255)           | yes      |         |              |
| corrupted   | boolean                          | yes      |         |              |
| quality     | integer                          | yes      |         |              |
| ilvl        | integer                          | yes      |         |              |
| created     | timestamp with time zone         | no       | now()   |              |
| updated     | timestamp with time zone         | no       | now()   |              |

Indexes: PK (id), confidence, date, itemId, (itemId, date), leagueId, (leagueId, date), (leagueId, itemId, date), UNIQUE (leagueId, itemId, date, variant, links), links, variant

---

## Table: items
| Column   | Type                    | Nullable | Default                | Description |
|----------|-------------------------|----------|------------------------|-------------|
| id       | integer                 | no       | nextval('items_id_seq'::regclass) | PK |
| name     | character varying(255)  | no       |                        | Indexed     |
| type     | character varying(255)  | yes      |                        |             |
| baseType | character varying(255)  | yes      |                        |             |
| variant  | character varying(255)  | yes      |                        |             |
| links    | character varying(255)  | yes      |                        |             |
| created  | timestamp with time zone| no       | now()                  |             |
| updated  | timestamp with time zone| no       | now()                  |             |

Indexes: PK (id), base_type, category, name, type

---

## Table: predictions
| Column         | Type                              | Nullable | Default | Description |
|----------------|-----------------------------------|----------|---------|-------------|
| id             | integer                           | no       |         | PK          |
| currencyId     | integer                           | yes      |         | FK → currency(id) |
| predictionType | enum_predictions_predictionType   | no       |         |              |
| timeframe      | enum_predictions_timeframe        | no       |         |              |
| currentPrice   | numeric(10,0)                     | no       |         |              |
| predictedPrice | numeric(10,0)                     | no       |         |              |
| confidence     | numeric(5,0)                      | no       |         | 0-1 score   |
| trend          | enum_predictions_trend            | no       |         |              |
| volatility     | numeric(5,4)                      | yes      |         |              |
| created        | timestamp with time zone          | no       | now()   |              |
| updated        | timestamp with time zone          | no       | now()   |              |

Indexes: PK (id), confidence, currencyId, timeframe, trend

---

## Relationships
- **currency_prices**: FK to leagues, currency (get/pay)
- **item_prices**: FK to leagues, items
- **predictions**: FK to currency

---

This document provides a full reference for the PoEconomy database schema, including all tables, columns, types, constraints, indexes, and relationships. Use this as a guide for data integration, querying, and further development. 