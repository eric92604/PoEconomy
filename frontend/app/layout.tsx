import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { QueryProvider, ThemeProvider } from "@/lib/providers";
import { Header, Footer } from "@/components/layout";
import { APP_NAME } from "@/lib/constants/config";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
});

export const metadata: Metadata = {
  title: {
    default: `${APP_NAME} - Path of Exile Currency Predictions`,
    template: `%s | ${APP_NAME}`,
  },
  description: "Real-time currency price predictions and investment analysis for Path of Exile",
  keywords: [
    "Path of Exile",
    "POE",
    "currency",
    "predictions",
    "investment",
    "trading",
    "market analysis",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <ThemeProvider defaultTheme="dark" storageKey="poe-theme">
          <QueryProvider>
            <div className="relative flex min-h-screen flex-col">
              <Header />
              <main className="flex-1 w-full">
                <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
                  {children}
                </div>
              </main>
              <Footer />
            </div>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
