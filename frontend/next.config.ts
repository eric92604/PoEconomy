import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  
  // Enable experimental features
  experimental: {
    // Optimize package imports
    optimizePackageImports: ["lucide-react", "recharts", "@tanstack/react-query"],
  },

  // Image optimization
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "web.poecdn.com",
        pathname: "/**",
      },
    ],
  },

  // Environment variables validation
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_ENVIRONMENT: process.env.NEXT_PUBLIC_ENVIRONMENT,
  },
};

export default nextConfig;
