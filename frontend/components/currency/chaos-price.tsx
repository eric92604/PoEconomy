import { CurrencyIcon } from "./currency-icon";

interface ChaosPriceProps {
  price: number;
  className?: string;
}

export function ChaosPrice({ price, className }: ChaosPriceProps) {
  const formatPrice = (value: number): string => {
    if (value >= 1000) {
      return (value / 1000).toFixed(1) + 'k';
    }
    return value.toFixed(2);
  };

  return (
    <span className={`inline-flex items-center gap-1 ${className || ''}`}>
      <span>{formatPrice(price)}</span>
      <CurrencyIcon 
        iconUrl="/images/chaos-orb.png" 
        currency="Chaos Orb" 
        size="sm" 
      />
    </span>
  );
}
