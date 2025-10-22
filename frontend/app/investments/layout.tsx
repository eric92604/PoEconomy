import type { Metadata } from "next";

export const metadata: Metadata = {
  title: 'Investment Opportunities - Currency Trading Analysis',
  description: 'Discover the most profitable Path of Exile currency investments with AI-powered predictions. Analyze short-term, medium-term, and long-term investment opportunities.',
  keywords: [
    'Path of Exile investments',
    'POE currency investments',
    'currency trading opportunities',
    'POE investment analysis',
    'profitable currency trades',
    'POE trading opportunities',
    'currency investment guide',
    'POE market predictions',
    'investment calculator',
    'trading analysis'
  ],
  openGraph: {
    title: 'Investment Opportunities - PoEconomy Trading Analysis',
    description: 'Discover the most profitable Path of Exile currency investments with AI-powered predictions and analysis.',
    url: 'https://poeconomy.com/investments',
    images: [
      {
        url: 'https://poeconomy.com/og-investments.png',
        width: 1200,
        height: 630,
        alt: 'PoEconomy Investment Opportunities - Currency Trading Analysis',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Investment Opportunities - PoEconomy Trading Analysis',
    description: 'Discover the most profitable Path of Exile currency investments with AI-powered predictions and analysis.',
    images: ['https://poeconomy.com/og-investments.png'],
  },
  alternates: {
    canonical: 'https://poeconomy.com/investments',
  },
};

export default function InvestmentsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
