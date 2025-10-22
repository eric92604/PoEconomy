import type { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Live Currency Prices - Real-time POE Market Data',
  description: 'Track real-time Path of Exile currency prices with live updates. Monitor price changes, trends, and market data across all leagues.',
  keywords: [
    'Path of Exile prices',
    'POE currency prices',
    'live currency prices',
    'POE market prices',
    'currency price tracker',
    'POE price monitoring',
    'real-time prices',
    'currency trends',
    'POE market data',
    'price analysis'
  ],
  openGraph: {
    title: 'Live Currency Prices - PoEconomy Real-time Market Data',
    description: 'Track real-time Path of Exile currency prices with live updates and market trends.',
    url: 'https://poeconomy.com/prices',
    images: [
      {
        url: 'https://poeconomy.com/og-prices.png',
        width: 1200,
        height: 630,
        alt: 'PoEconomy Live Currency Prices - Real-time Market Data',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Live Currency Prices - PoEconomy Real-time Market Data',
    description: 'Track real-time Path of Exile currency prices with live updates and market trends.',
    images: ['https://poeconomy.com/og-prices.png'],
  },
  alternates: {
    canonical: 'https://poeconomy.com/prices',
  },
};

export default function PricesLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
