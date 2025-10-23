import Link from "next/link";
import { TrendingUp, BarChart3, Target, Activity } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="py-12 space-y-12">
      {/* Hero Section */}
      <section className="flex flex-col items-center text-center space-y-6 py-12">
        <div className="flex items-center justify-center space-x-3">
          <TrendingUp className="h-16 w-16 text-primary" />
        </div>
        <h1 className="text-5xl font-bold tracking-tight sm:text-6xl md:text-7xl">
          PoEconomy - Path of Exile
          <br />
          <span className="text-primary">Market Intelligence</span>
        </h1>
        <p className="max-w-[700px] text-lg text-muted-foreground sm:text-xl">
          Real-time currency price predictions and investment analysis.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <div className="flex gap-4">
            <Button asChild size="lg">
              <Link href="/dashboard">View Dashboard</Link>
            </Button>
            <Button asChild variant="outline" size="lg">
              <Link href="/investments">Explore Investments</Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <BarChart3 className="h-8 w-8 mb-2 text-primary" />
            <CardTitle>Real-Time Predictions</CardTitle>
            <CardDescription>
              AI-powered predictions for 1-day, 3-day, and 7-day price movements
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Get accurate price predictions with confidence scores to make better trading decisions.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Target className="h-8 w-8 mb-2 text-primary" />
            <CardTitle>Investment Opportunities</CardTitle>
            <CardDescription>
              Discover the most profitable short-term and long-term investments
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Filter and sort currencies by profit potential, confidence, and risk level.
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <Activity className="h-8 w-8 mb-2 text-primary" />
            <CardTitle>Market Intelligence</CardTitle>
            <CardDescription>
              Track historical prices and market trends across all leagues
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">
              Advanced charting and analytics to help you understand market dynamics.
            </p>
          </CardContent>
        </Card>
      </section>

    </div>
  );
}
