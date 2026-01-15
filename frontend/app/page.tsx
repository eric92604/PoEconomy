import type { Metadata } from "next";
import Link from "next/link";
import Image from "next/image";
import { ArrowRight, TrendingUp, Shield, BarChart3, Zap } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { LiveStatsBar } from "@/components/landing/live-stats-bar";

export const metadata: Metadata = {
  alternates: {
    canonical: 'https://poeconomy.com',
  },
};

export default function Home() {
  return (
    <div className="relative">
      {/* Hero Section */}
      <section className="relative hero-gradient py-12 sm:py-16">
        <div className="flex flex-col items-center text-center space-y-6">
          {/* Mirror of Kalandra Hero Visual */}
          <div className="relative">
            <div className="absolute inset-0 bg-gradient-to-b from-[var(--poe-gold)]/20 to-transparent blur-3xl scale-150" />
            <div className="relative mirror-glow">
              <Image
                src="/images/mirror-of-kalandra.png"
                alt="Mirror of Kalandra"
                width={120}
                height={120}
                className="drop-shadow-2xl"
                priority
              />
            </div>
            {/* Decorative price prediction arrow */}
            <div className="absolute -right-16 top-1/2 -translate-y-1/2 hidden sm:flex items-center gap-2 text-poe-gold">
              <ArrowRight className="h-6 w-6 animate-pulse" />
              <span className="text-sm font-mono bg-card/80 px-2 py-1 rounded border border-poe-gold/30">
                +12.4%
              </span>
            </div>
          </div>

          {/* Headline */}
          <div className="space-y-4 max-w-4xl">
            <h1 className="text-4xl font-bold tracking-tight sm:text-5xl md:text-6xl lg:text-7xl">
              Your Edge in{" "}
              <span className="text-poe-gold">Wraeclast&apos;s Economy</span>
            </h1>
            <p className="max-w-[700px] mx-auto text-lg text-muted-foreground sm:text-xl">
              ML-powered price predictions for Divine Orbs, Mirrors, and 120+ currencies.
              <br className="hidden sm:block" />
              <span className="text-foreground/80">Stop guessing. Start profiting.</span>
            </p>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-4 items-center pt-4">
            <Button asChild size="lg" className="bg-poe-gold hover:bg-poe-gold/90 text-background font-semibold px-8 glow-gold-sm">
              <Link href="/investments">
                See Top Gainers
                <TrendingUp className="ml-2 h-4 w-4" />
              </Link>
            </Button>
            <Button asChild variant="outline" size="lg" className="border-poe-gold/50 hover:border-poe-gold hover:bg-poe-gold/10">
              <Link href="/dashboard">
                View Dashboard
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Live Stats Bar */}
      <LiveStatsBar />

      {/* Features Section */}
      <section className="py-10 sm:py-12">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold sm:text-3xl mb-3">
            Why Exiles Trust <span className="text-poe-gold">PoEconomy</span>
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built by traders, for traders. Get the intelligence edge in every league.
          </p>
        </div>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {/* Feature 1 - Predictions */}
          <Card className="border-gradient-gold bg-card/50 backdrop-blur-sm hover:glow-gold-sm transition-shadow duration-300">
            <CardHeader className="relative">
              <div className="absolute top-4 right-4 opacity-10">
                <Image
                  src="/images/divine-orb.png"
                  alt=""
                  width={48}
                  height={48}
                  className="opacity-50"
                />
              </div>
              <div className="h-10 w-10 rounded-lg bg-poe-gold/10 flex items-center justify-center mb-3">
                <BarChart3 className="h-5 w-5 text-poe-gold" />
              </div>
              <CardTitle className="text-xl">Predict Divine Crashes</CardTitle>
              <CardDescription className="text-muted-foreground">
                1-day, 3-day, and 7-day price forecasts with confidence scores
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Know when to hold and when to flip. Our ML models analyze historical patterns to predict price movements before they happen.
              </p>
            </CardContent>
          </Card>

          {/* Feature 2 - Speed */}
          <Card className="border-gradient-gold bg-card/50 backdrop-blur-sm hover:glow-gold-sm transition-shadow duration-300">
            <CardHeader className="relative">
              <div className="absolute top-4 right-4 opacity-10">
                <Image
                  src="/images/chaos-orb.png"
                  alt=""
                  width={48}
                  height={48}
                  className="opacity-50"
                />
              </div>
              <div className="h-10 w-10 rounded-lg bg-poe-gold/10 flex items-center justify-center mb-3">
                <Zap className="h-5 w-5 text-poe-gold" />
              </div>
              <CardTitle className="text-xl">League-Start Ready</CardTitle>
              <CardDescription className="text-muted-foreground">
                Fresh predictions every day, optimized for early economy chaos
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Hit the ground running each league. Our models adapt to volatile early-league economies when the biggest profits are made.
              </p>
            </CardContent>
          </Card>

          {/* Feature 3 - Confidence */}
          <Card className="border-gradient-gold bg-card/50 backdrop-blur-sm hover:glow-gold-sm transition-shadow duration-300">
            <CardHeader className="relative">
              <div className="absolute top-4 right-4 opacity-10">
                <Image
                  src="/images/mirror-of-kalandra.png"
                  alt=""
                  width={48}
                  height={48}
                  className="opacity-50"
                />
              </div>
              <div className="h-10 w-10 rounded-lg bg-poe-gold/10 flex items-center justify-center mb-3">
                <Shield className="h-5 w-5 text-poe-gold" />
              </div>
              <CardTitle className="text-xl">Confidence Scores</CardTitle>
              <CardDescription className="text-muted-foreground">
                Know how certain each prediction is before you invest
              </CardDescription>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Every prediction includes a confidence rating. Filter by high-confidence plays or take calculated risks on volatile currencies.
              </p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-10 sm:py-12 border-t border-border/40">
        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold sm:text-3xl mb-3">
            How It Works
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            From raw market data to actionable predictions in three steps
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="text-center space-y-3">
            <div className="h-12 w-12 rounded-full bg-poe-gold/10 border border-poe-gold/30 flex items-center justify-center mx-auto">
              <span className="text-poe-gold font-bold">1</span>
            </div>
            <h3 className="font-semibold text-lg">Data Ingestion</h3>
            <p className="text-sm text-muted-foreground">
              We pull real-time price data every hour, tracking 120+ currencies across all active leagues.
            </p>
          </div>

          <div className="text-center space-y-3">
            <div className="h-12 w-12 rounded-full bg-poe-gold/10 border border-poe-gold/30 flex items-center justify-center mx-auto">
              <span className="text-poe-gold font-bold">2</span>
            </div>
            <h3 className="font-semibold text-lg">ML Analysis</h3>
            <p className="text-sm text-muted-foreground">
              Our ensemble models analyze price history, volume trends, and market patterns to generate predictions.
            </p>
          </div>

          <div className="text-center space-y-3">
            <div className="h-12 w-12 rounded-full bg-poe-gold/10 border border-poe-gold/30 flex items-center justify-center mx-auto">
              <span className="text-poe-gold font-bold">3</span>
            </div>
            <h3 className="font-semibold text-lg">Actionable Intel</h3>
            <p className="text-sm text-muted-foreground">
              Browse predictions by timeframe, filter by confidence, and find the flip targets that match your risk tolerance.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}
