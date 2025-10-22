// Auto-generated currency icon mapping
export const CURRENCY_ICON_MAP = {
  "exalted_orb": "/images/currency/exalted-orb.png",
  "mirror_of_kalandra": "/images/currency/mirror-of-kalandra.png",
  "divine_orb": "/images/currency/divine-orb.png",
  "orb_of_fusing": "/images/currency/orb-of-fusing.png",
  "chaos_orb": "/images/currency/chaos-orb.png",
  "orb_of_regret": "/images/currency/orb-of-regret.png",
  "chromatic_orb": "/images/currency/chromatic-orb.png",
  "orb_of_scouring": "/images/currency/orb-of-scouring.png",
  "orb_of_alchemy": "/images/currency/orb-of-alchemy.png",
  "vaal_orb": "/images/currency/vaal-orb.png",
  "jeweller's_orb": "/images/currency/jewellers-orb.png",
  "orb_of_chance": "/images/currency/orb-of-chance.png",
  "orb_of_alteration": "/images/currency/orb-of-alteration.png",
  "fusing_orb": "/images/currency/fusing-orb.png",
  "orb_of_horizons": "/images/currency/orb-of-horizons.png",
  "orb_of_binding": "/images/currency/orb-of-binding.png",
  "orb_of_augmentation": "/images/currency/orb-of-augmentation.png",
  "orb_of_transmutation": "/images/currency/orb-of-transmutation.png",
  "blessing_of_xoph": "/images/currency/blessing-of-xoph.png",
  "blessing_of_esh": "/images/currency/blessing-of-esh.png",
  "blessing_of_chayula": "/images/currency/blessing-of-chayula.png",
  "blessing_of_uul-netol": "/images/currency/blessing-of-uul-netol.png",
  "blessing_of_tul": "/images/currency/blessing-of-tul.png",
  "ancient_orb": "/images/currency/ancient-orb.png",
  "annulment_orb": "/images/currency/annulment-orb.png",
  "eternal_orb": "/images/currency/eternal-orb.png"
};

export function getCurrencyIconPath(currencyName: string): string | undefined {
  const key = currencyName.toLowerCase().replace(/\s+/g, '_');
  return CURRENCY_ICON_MAP[key];
}
