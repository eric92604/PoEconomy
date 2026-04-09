import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "League History - PoEconomy",
  description:
    "Compare Path of Exile currency prices across previous leagues. Overlay historical price charts to identify recurring market patterns.",
};

export default function LeagueHistoryLayout({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
