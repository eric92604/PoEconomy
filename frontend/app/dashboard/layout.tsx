import type { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Dashboard - Market Overview & Key Metrics',
  description: 'Real-time Path of Exile market overview with key metrics, top gainers, and market intelligence. Track currency performance and market trends.',
  keywords: [
    'Path of Exile dashboard',
    'POE market overview', 
    'currency metrics',
    'POE trading dashboard',
    'market intelligence',
    'currency performance',
    'POE market trends',
    'trading analytics'
  ],
  openGraph: {
    title: 'Dashboard - PoEconomy Market Overview',
    description: 'Real-time Path of Exile market overview with key metrics, top gainers, and market intelligence.',
    url: 'https://poeconomy.com/dashboard',
    images: [
      {
        url: 'https://poeconomy.com/og-dashboard.png',
        width: 1200,
        height: 630,
        alt: 'PoEconomy Dashboard - Market Overview',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Dashboard - PoEconomy Market Overview',
    description: 'Real-time Path of Exile market overview with key metrics, top gainers, and market intelligence.',
    images: ['https://poeconomy.com/og-dashboard.png'],
  },
  alternates: {
    canonical: 'https://poeconomy.com/dashboard',
  },
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
