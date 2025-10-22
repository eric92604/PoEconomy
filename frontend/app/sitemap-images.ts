import { MetadataRoute } from 'next'

export default function sitemapImages(): MetadataRoute.Sitemap {
  const baseUrl = 'https://poeconomy.com'
  const currentDate = new Date()
  
  return [
    {
      url: `${baseUrl}/og-image.png`,
      lastModified: currentDate,
      changeFrequency: 'monthly',
      priority: 0.8,
    },
    {
      url: `${baseUrl}/og-dashboard.png`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/og-investments.png`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.7,
    },
    {
      url: `${baseUrl}/og-prices.png`,
      lastModified: currentDate,
      changeFrequency: 'weekly',
      priority: 0.7,
    },
  ]
}
