export const APP_NAME = "PoEconomy";

export const APP_DESCRIPTION = "Real-time currency price predictions and investment analysis for Path of Exile";

export const APP_VERSION = "1.0.0";

export const SUPPORTED_LEAGUES = [
  "Affliction",
  "Ancestor",
  "Keepers",
  "Mercenaries",
  "Necro Settlers",
  "Necropolis",
  "Phrecia",
  "Settlers"
] as const;

export const DEFAULT_LEAGUE = "Affliction" as const;

export const CURRENCY_GROUPS = {
  CURRENCY: "Currency",
  FRAGMENTS: "Fragments", 
  DIVINATION_CARDS: "Divination Cards",
  ESSENCES: "Essences",
  FOSSILS: "Fossils",
  RESONATORS: "Resonators",
  SCARABS: "Scarabs",
  DELVE: "Delve",
  INCURSION: "Incursion",
  HARVEST: "Harvest",
  METAMORPHS: "Metamorphs",
  OILS: "Oils",
  CATALYSTS: "Catalysts",
  VIALS: "Vials",
  NETS: "Nets",
  LEGION: "Legion",
  BLIGHT: "Blight",
  DELIRIUM: "Delirium",
  HEIST: "Heist",
  RITUAL: "Ritual",
  ULTIMATUM: "Ultimatum",
  EXPEDITION: "Expedition",
  TATTOOS: "Tattoos",
  OMENS: "Omens",
  VOIDSTONES: "Voidstones",
  PRISMATIC: "Prismatic",
  MEMORIES: "Memories",
  INVITATIONS: "Invitations",
  BLIGHTED: "Blighted",
  BLIGHT_RAVAGED: "Blight Ravaged",
  UNIQUE: "Unique",
  UNIQUE_FRAGMENTS: "Unique Fragments",
  UNIQUE_RELICS: "Unique Relics",
  UNIQUE_JEWELS: "Unique Jewels",
  UNIQUE_FLASKS: "Unique Flasks",
  UNIQUE_WEAPONS: "Unique Weapons",
  UNIQUE_ARMOUR: "Unique Armour",
  UNIQUE_ACCESSORIES: "Unique Accessories",
  UNIQUE_MAPS: "Unique Maps",
  UNIQUE_PROPHECIES: "Unique Prophecies",
  UNIQUE_BEASTS: "Unique Beasts",
  UNIQUE_ABYSS: "Unique Abyss",
  UNIQUE_DELVE: "Unique Delve",
  UNIQUE_INCURSION: "Unique Incursion",
  UNIQUE_HARVEST: "Unique Harvest",
  UNIQUE_METAMORPHS: "Unique Metamorphs",
  UNIQUE_OILS: "Unique Oils",
  UNIQUE_CATALYSTS: "Unique Catalysts",
  UNIQUE_VIALS: "Unique Vials",
  UNIQUE_NETS: "Unique Nets",
  UNIQUE_LEGION: "Unique Legion",
  UNIQUE_BLIGHT: "Unique Blight",
  UNIQUE_DELIRIUM: "Unique Delirium",
  UNIQUE_HEIST: "Unique Heist",
  UNIQUE_RITUAL: "Unique Ritual",
  UNIQUE_ULTIMATUM: "Unique Ultimatum",
  UNIQUE_EXPEDITION: "Unique Expedition",
  UNIQUE_TATTOOS: "Unique Tattoos",
  UNIQUE_OMENS: "Unique Omens",
  UNIQUE_VOIDSTONES: "Unique Voidstones",
  UNIQUE_PRISMATIC: "Unique Prismatic",
  UNIQUE_MEMORIES: "Unique Memories",
  UNIQUE_INVITATIONS: "Unique Invitations",
  UNIQUE_BLIGHTED: "Unique Blighted",
  UNIQUE_BLIGHT_RAVAGED: "Unique Blight Ravaged"
} as const;

export const THEME_CONFIG = {
  defaultTheme: "dark" as const,
  storageKey: "poe-theme" as const,
  themes: ["light", "dark"] as const
} as const;

export const API_CONFIG = {
  baseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || "https://api.poeconomy.com",
  timeout: 10000,
  retryAttempts: 3
} as const;

export const ENABLE_DEVTOOLS = process.env.NODE_ENV === "development";

export const STORAGE_KEYS = {
  THEME: "poe-theme",
  LEAGUE: "poe-league",
  CURRENCY_FILTERS: "poe-currency-filters",
  WATCHLIST: "poe-watchlist",
  PREFERENCES: "poe-preferences"
} as const;
