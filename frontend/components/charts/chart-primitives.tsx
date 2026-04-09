"use client";

/**
 * Shared chart primitives reused across price chart components.
 */

import { formatChaosPrice } from "@/lib/utils";

export const CustomYAxisTick = ({ x, y, payload, isDark }: any) => (
  <g transform={`translate(${x},${y})`}>
    <text
      x={-18}
      y={0}
      dy={4}
      textAnchor="end"
      fill={isDark ? "#9ca3af" : "#6b7280"}
      fontSize={12}
    >
      {formatChaosPrice(payload.value)}
    </text>
    <image x={-14} y={-6} width={12} height={12} href="/images/chaos-orb.png" />
  </g>
);
