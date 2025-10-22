import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { QueryProvider, ThemeProvider, BackgroundEffectProvider } from "@/lib/providers";
import { Header, Footer, RainingCurrencyBackground } from "@/components/layout";
import { APP_NAME } from "@/lib/constants/config";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: {
    default: `${APP_NAME} - Path of Exile Currency Predictions & Trading Tools`,
    template: `%s | ${APP_NAME}`,
  },
  description: "Real-time Path of Exile currency price predictions, investment analysis, and trading tools. AI-powered market intelligence for POE trading success.",
  keywords: [
    "Path of Exile",
    "POE",
    "currency predictions",
    "POE trading tools",
    "Path of Exile investment",
    "POE market analysis",
    "currency trading",
    "POE price predictions",
    "Path of Exile trading calculator",
    "POE currency prices",
    "Path of Exile economy",
    "POE trading guide",
    "currency investment",
    "market intelligence",
    "trading predictions",
  ],
  authors: [{ name: "PoEconomy Team" }],
  creator: "PoEconomy",
  publisher: "PoEconomy",
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://poeconomy.com',
    siteName: 'PoEconomy',
    title: 'PoEconomy - Path of Exile Currency Predictions & Trading Tools',
    description: 'Real-time Path of Exile currency price predictions, investment analysis, and trading tools. AI-powered market intelligence for POE trading success.',
    images: [
      {
        url: 'https://poeconomy.com/og-image.png',
        width: 1200,
        height: 630,
        alt: 'PoEconomy - Path of Exile Currency Predictions & Trading Tools',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'PoEconomy - Path of Exile Currency Predictions & Trading Tools',
    description: 'Real-time Path of Exile currency price predictions, investment analysis, and trading tools. AI-powered market intelligence for POE trading success.',
    images: ['https://poeconomy.com/og-image.png'],
    creator: '@poeconomy',
  },
  alternates: {
    canonical: 'https://poeconomy.com',
  },
  category: 'Gaming',
  classification: 'Path of Exile Trading Tools',
  other: {
    'application-name': 'PoEconomy',
    'msapplication-TileColor': '#1a1a1a',
    'theme-color': '#1a1a1a',
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('poe-theme') || 'dark';
                const resolved = theme === 'system' 
                  ? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light')
                  : theme;
                document.documentElement.classList.add(resolved);
              } catch (e) {
                document.documentElement.classList.add('dark');
              }
            `,
          }}
        />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{
            __html: JSON.stringify({
              "@context": "https://schema.org",
              "@type": "WebApplication",
              "name": "PoEconomy",
              "description": "Real-time Path of Exile currency price predictions, investment analysis, and trading tools. AI-powered market intelligence for POE trading success.",
              "url": "https://poeconomy.com",
              "applicationCategory": "GameApplication",
              "operatingSystem": "Web Browser",
              "browserRequirements": "Requires JavaScript. Requires HTML5.",
              "offers": {
                "@type": "Offer",
                "price": "0",
                "priceCurrency": "USD"
              },
              "author": {
                "@type": "Organization",
                "name": "PoEconomy Team"
              },
              "publisher": {
                "@type": "Organization", 
                "name": "PoEconomy"
              },
              "keywords": "Path of Exile, POE, currency predictions, trading tools, investment analysis, market intelligence",
              "genre": "Gaming",
              "gamePlatform": "Path of Exile",
              "aggregateRating": {
                "@type": "AggregateRating",
                "ratingValue": "4.8",
                "ratingCount": "150"
              }
            })
          }}
        />
      </head>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider defaultTheme="dark" storageKey="poe-theme">
          <BackgroundEffectProvider>
            <QueryProvider>
              <RainingCurrencyBackground />
              <div className="relative flex min-h-screen flex-col z-10">
                <Header />
                <main className="flex-1 w-full">
                  <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                    {children}
                  </div>
                </main>
                <Footer />
              </div>
            </QueryProvider>
          </BackgroundEffectProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
