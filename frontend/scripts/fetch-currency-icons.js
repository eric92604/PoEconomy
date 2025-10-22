/**
 * Script to fetch and bundle currency icons from PoE CDN
 * Downloads all currencies from the API and converts them to optimized AVIF format
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { createHash } = require('crypto');
const sharp = require('sharp');

// Configuration
const ICONS_DIR = path.join(__dirname, '../public/images/currency');
const ICON_SIZE = 32; // Standard PoE icon size
const TIMEOUT = 15000; // 15 second timeout per icon
const BATCH_SIZE = 100;
const DELAY_BETWEEN_BATCHES = 1000; // 2 second delay between batches

// Cloudflare Pages API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.poeconomy.com';
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

// Image optimization settings
const IMAGE_OPTIMIZATION = {
  format: 'avif', // AVIF provides best compression (~50% smaller than PNG)
  quality: 85, // Good balance between quality and file size
  effort: 9, // Maximum compression effort (0-9)
  maxWidth: 64, // 2x the display size for retina support
  maxHeight: 64,
};

// Create directory if it doesn't exist
if (!fs.existsSync(ICONS_DIR)) {
  fs.mkdirSync(ICONS_DIR, { recursive: true });
}

/**
 * Fetch all currencies from the API
 */
async function fetchAllCurrencies() {
  console.log('🔍 Fetching currency metadata from API...');
  
  const url = `${API_BASE_URL}/predict/currencies`;
  const options = {
    headers: {
      'Accept': 'application/json',
      'User-Agent': 'PoEconomy-IconFetcher/1.0',
      ...(API_KEY && { 'Authorization': `Bearer ${API_KEY}` })
    }
  };

  return new Promise((resolve, reject) => {
    const request = https.get(url, options, (response) => {
      let data = '';
      
      response.on('data', (chunk) => {
        data += chunk;
      });
      
      response.on('end', () => {
        try {
          const parsed = JSON.parse(data);
          if (parsed.currencies) {
            resolve(parsed.currencies);
          } else {
            reject(new Error('Invalid API response format'));
          }
        } catch (error) {
          reject(new Error(`Failed to parse API response: ${error.message}`));
        }
      });
    });

    request.on('error', (error) => {
      reject(new Error(`API request failed: ${error.message}`));
    });

    request.setTimeout(TIMEOUT, () => {
      request.destroy();
      reject(new Error('API request timeout'));
    });
  });
}

/**
 * Extract unique currency names and icon URLs from API response
 */
function extractCurrencyData(currencyData) {
  const currencies = new Map();
  
  Object.entries(currencyData).forEach(([currencyName, leagueData]) => {
    Object.values(leagueData).forEach((metadata) => {
      if (metadata.icon_url && !currencies.has(currencyName)) {
        currencies.set(currencyName, {
          name: currencyName,
          iconUrl: metadata.icon_url,
          category: metadata.category || 'currency'
        });
      }
    });
  });
  
  return Array.from(currencies.values());
}

/**
 * Convert PNG to AVIF format with optimization
 */
async function convertToAvif(pngPath, avifPath) {
  try {
    await sharp(pngPath)
      .avif({
        quality: 85,
        effort: 9, // Maximum compression effort
        chromaSubsampling: '4:4:4', // Best quality
      })
      .toFile(avifPath);
    return true;
  } catch (error) {
    console.error(`Failed to convert ${pngPath} to AVIF:`, error.message);
    return false;
  }
}

/**
 * Download a single icon from the provided URL and convert to AVIF
 */
function downloadIcon(currencyName, iconUrl, retries = 3) {
  return new Promise((resolve, reject) => {
    // Create safe filename
    const safeName = currencyName
      .toLowerCase()
      .replace(/[^a-z0-9]/g, '_')
      .replace(/_+/g, '_')
      .replace(/^_|_$/g, '');
    
    const pngFilename = `${safeName}.png`;
    const avifFilename = `${safeName}.avif`;
    const pngFilepath = path.join(ICONS_DIR, pngFilename);
    const avifFilepath = path.join(ICONS_DIR, avifFilename);
    
    // Check if AVIF file already exists
    if (fs.existsSync(avifFilepath)) {
      console.log(`✓ ${currencyName} (already exists)`);
      resolve({ currencyName, filename: avifFilename, filepath: avifFilepath, iconUrl });
      return;
    }

    const file = fs.createWriteStream(pngFilepath);
    
    const request = https.get(iconUrl, (response) => {
      if (response.statusCode === 200) {
        response.pipe(file);
        file.on('finish', async () => {
          file.close();
          console.log(`✓ ${currencyName} downloaded`);
          
          // Convert to AVIF
          const converted = await convertToAvif(pngFilepath, avifFilepath);
          
          if (converted) {
            // Delete the original PNG to save space
            fs.unlinkSync(pngFilepath);
            resolve({ currencyName, filename: avifFilename, filepath: avifFilepath, iconUrl });
          } else {
            // Keep PNG if AVIF conversion failed
            console.log(`  ⚠️  Using PNG fallback`);
            resolve({ currencyName, filename: pngFilename, filepath: pngFilepath, iconUrl });
          }
        });
      } else if (response.statusCode === 404 && retries > 0) {
        console.log(`⚠️  ${currencyName} not found, retrying... (${retries} attempts left)`);
        setTimeout(() => {
          downloadIcon(currencyName, iconUrl, retries - 1)
            .then(resolve)
            .catch(reject);
        }, 2000);
      } else {
        reject(new Error(`HTTP ${response.statusCode}: ${currencyName}`));
      }
    });

    request.on('error', (err) => {
      fs.unlink(pngFilepath, () => {}); // Delete partial file
      if (retries > 0) {
        console.log(`Retrying ${currencyName}... (${retries} attempts left)`);
        setTimeout(() => {
          downloadIcon(currencyName, iconUrl, retries - 1)
            .then(resolve)
            .catch(reject);
        }, 2000);
      } else {
        reject(err);
      }
    });

    request.setTimeout(TIMEOUT, () => {
      request.destroy();
      fs.unlink(pngFilepath, () => {}); // Delete partial file
      reject(new Error(`Timeout: ${currencyName}`));
    });
  });
}

/**
 * Generate optimized icon mapping file with Cloudflare optimizations
 */
function generateIconMapping(downloadedIcons) {
  const mapping = {};
  
  downloadedIcons.forEach(({ currencyName, filename }) => {
    const key = currencyName.toLowerCase().replace(/\s+/g, '_');
    
    mapping[key] = `/images/currency/${filename}`;
  });

  const mappingContent = `// Auto-generated currency icon mapping
export const CURRENCY_ICON_MAP: Record<string, string> = ${JSON.stringify(mapping, null, 2)};

export function getCurrencyIconPath(currencyName: string): string | undefined {
  const key = currencyName.toLowerCase().replace(/\\s+/g, '_');
  return CURRENCY_ICON_MAP[key];
}

// Helper function (currently returns PNG, can be extended for WebP/AVIF when conversion is implemented)
export function getOptimizedCurrencyIcon(currencyName: string): string | undefined {
  return getCurrencyIconPath(currencyName);
}
`;

  fs.writeFileSync(
    path.join(__dirname, '../lib/constants/currency-icons.ts'),
    mappingContent
  );
  
  console.log('✓ Optimized icon mapping generated');
}

/**
 * Main execution
 */
async function main() {
  console.log('🚀 Starting currency icon download from API...');
  console.log(`📁 Saving to: ${ICONS_DIR}`);
  console.log(`🌐 API URL: ${API_BASE_URL}`);
  
  try {
    // Fetch all currencies from API
    const currencyData = await fetchAllCurrencies();
    const currencies = extractCurrencyData(currencyData);
    
    console.log(`📋 Found ${currencies.length} currencies in API`);
    
    const downloadedIcons = [];
    const failedIcons = [];
    
    // Download icons in small batches to avoid rate limiting
    for (let i = 0; i < currencies.length; i += BATCH_SIZE) {
      const batch = currencies.slice(i, i + BATCH_SIZE);
      const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
      const totalBatches = Math.ceil(currencies.length / BATCH_SIZE);
      
      console.log(`\n📦 Processing batch ${batchNumber}/${totalBatches} (${batch.length} currencies)`);
      
      const promises = batch.map(async (currency) => {
        try {
          const result = await downloadIcon(currency.name, currency.iconUrl);
          downloadedIcons.push(result);
          return result;
        } catch (error) {
          console.log(`❌ Failed to download ${currency.name}: ${error.message}`);
          failedIcons.push({ currency: currency.name, error: error.message });
          return null;
        }
      });
      
      await Promise.all(promises);
      
      // Delay between batches to be respectful to the CDN
      if (i + BATCH_SIZE < currencies.length) {
        console.log(`⏳ Waiting ${DELAY_BETWEEN_BATCHES}ms before next batch...`);
        await new Promise(resolve => setTimeout(resolve, DELAY_BETWEEN_BATCHES));
      }
    }
    
    // Generate optimized mapping file
    generateIconMapping(downloadedIcons);
    
    // Summary
    console.log('\n📊 Download Summary:');
    console.log(`✅ Successfully downloaded: ${downloadedIcons.length} icons`);
    console.log(`❌ Failed downloads: ${failedIcons.length} icons`);
    console.log(`📈 Success rate: ${Math.round((downloadedIcons.length / currencies.length) * 100)}%`);
    
    if (failedIcons.length > 0) {
      console.log('\n❌ Failed icons:');
      failedIcons.forEach(({ currency, error }) => {
        console.log(`  - ${currency}: ${error}`);
      });
    }
    
    console.log('\n🎉 Currency icon bundling complete!');
    console.log('💡 Icons are now available in optimized AVIF format.');
    console.log('📊 AVIF provides ~50% smaller file sizes compared to PNG with same quality.');
    console.log('🚀 Next steps:');
    console.log('   1. Run "npm run build" to include icons in deployment');
    console.log('   2. Deploy to Cloudflare Pages');
    console.log('   3. Icons will load faster due to smaller file sizes');
    
  } catch (error) {
    console.error('❌ Failed to fetch currencies from API:', error.message);
    console.log('\n💡 Troubleshooting:');
    console.log('   - Check that NEXT_PUBLIC_API_URL is set correctly');
    console.log('   - Verify API is accessible and returning currency data');
    console.log('   - Check network connectivity');
    process.exit(1);
  }
}

// Run the script
if (require.main === module) {
  main().catch(console.error);
}

module.exports = { downloadIcon, generateIconMapping };
