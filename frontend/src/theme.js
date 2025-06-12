import { createSystem, defaultConfig } from "@chakra-ui/react"

// Define the theme configuration
const themeConfig = {
  theme: {
    tokens: {
      colors: {
        brand: {
          50: { value: '#E8EDE9' },  // Very light forest green
          100: { value: '#D1DCD4' },  // Light forest green
          200: { value: '#B9CABE' },  // Lighter forest green
          300: { value: '#A2B8A8' },  // Light forest green
          400: { value: '#8BA792' },  // Medium forest green
          500: { value: '#3E5641' },  // Forest green - Button/CTA
          600: { value: '#37493A' },  // Deeper forest green
          700: { value: '#2F3D32' },  // Dark forest green
          800: { value: '#283029' },  // Very dark forest green
          900: { value: '#1F2420' },  // Extremely dark forest green
        },
        teal:{
          50: { value: '#E6F7F2' },  // Very light teal
          100: { value: '#BFECE0' },  // Light teal
          200: { value: '#99E0D0' },  // Lighter teal
          300: { value: '#73D4C1' },  // Light teal
          400: { value: '#4DC8B2' },  // Medium teal
          500: { value: '#27BDA3' },  // Teal - Button/CTA
          600: { value: '#1F9E8A' },  // Deeper teal
          700: { value: '#177F71' },  // Dark teal
          800: { value: '#0F5F58' },  // Very dark teal
          900: { value: '#073F3F' },  // Extremely dark teal
        },
        ruby: {
          50: { value: '#FCE4EC' },  // Very light ruby
          100: { value: '#F8BBD0' },  // Light ruby
          200: { value: '#F48FB1' },  // Lighter ruby
          300: { value: '#F06292' },  // Light ruby
          400: { value: '#EC407A' },  // Medium ruby
          500: { value: '#E91E63' },  // Ruby - Button/CTA
          600: { value: '#D81B60' },  // Deeper ruby
          700: { value: '#C2185B' },  // Dark ruby
          800: { value: '#AD1457' },  // Very dark ruby
          900: { value: '#880E4F' },  // Extremely dark ruby
        },
        accent: {
          teal: { value: '#004D61' },    // Dark teal - Accent 1
          ruby: { value: '#822659' },    // Deep ruby - Accent 2
        },
        voltage: {
          low: { value: '#004D61' },    // Using Accent 1 - dark teal for low voltage
          medium: { value: '#822659' }, // Using Accent 2 - deep ruby for medium voltage
          high: { value: '#A32D6E' },   // Brighter ruby for high voltage
          danger: { value: '#C83483' }, // Vibrant ruby for danger
        },
        // Semantic colors using the new palette
        semantic: {
          background: { value: { base: '#FFFFFF', _dark: '#1A1A1A' } }, // Rich black for dark mode
          surface: { value: { base: '#F8F8F8', _dark: '#262626' } },    // Slightly lighter black
          muted: { value: { base: '#F0F0F0', _dark: '#333333' } },      // Even lighter black
          border: { value: { base: '#E2E2E2', _dark: '#404040' } },     // Medium gray for borders
          text: { value: { base: '#333333', _dark: '#F0F0F0' } },       // Off-white for dark mode text
          textMuted: { value: { base: '#767676', _dark: '#B0B0B0' } },  // Medium gray for muted text
        }
      },
      // ... rest of tokens remain the same
      fonts: {
        heading: { value: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' },
        body: { value: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' },
      },
      shadows: {
        // ... existing shadow definitions
        xs: { value: { base: '0 1px 2px rgba(0, 0, 0, 0.05)', _dark: '0 1px 2px rgba(0, 0, 0, 0.3)' } },
        sm: { value: { base: '0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06)', _dark: '0 1px 3px rgba(0, 0, 0, 0.4), 0 1px 2px rgba(0, 0, 0, 0.3)' } },
        md: { value: { base: '0 4px 6px rgba(0, 0, 0, 0.1), 0 2px 4px rgba(0, 0, 0, 0.06)', _dark: '0 4px 6px rgba(0, 0, 0, 0.4), 0 2px 4px rgba(0, 0, 0, 0.3)' } },
        lg: { value: { base: '0 10px 15px rgba(0, 0, 0, 0.1), 0 4px 6px rgba(0, 0, 0, 0.05)', _dark: '0 10px 15px rgba(0, 0, 0, 0.5), 0 4px 6px rgba(0, 0, 0, 0.4)' } },
        xl: { value: { base: '0 20px 25px rgba(0, 0, 0, 0.1), 0 10px 10px rgba(0, 0, 0, 0.04)', _dark: '0 20px 25px rgba(0, 0, 0, 0.5), 0 10px 10px rgba(0, 0, 0, 0.4)' } },
        '2xl': { value: { base: '0 25px 50px rgba(0, 0, 0, 0.25)', _dark: '0 25px 50px rgba(0, 0, 0, 0.7)' } },
        outline: { value: { base: '0 0 0 3px rgba(62, 86, 65, 0.6)', _dark: '0 0 0 3px rgba(62, 86, 65, 0.6)' } }, // Using forest green for focus ring
        inner: { value: { base: 'inset 0 2px 4px rgba(0, 0, 0, 0.06)', _dark: 'inset 0 2px 4px rgba(0, 0, 0, 0.3)' } },
      }
    },
    components: {
      Button: {
        defaultProps: {
          colorScheme: 'brand',
        },
        baseStyle: {
          boxShadow: 'sm', // Apply a subtle shadow by default
        },
      },
      Card: {
        baseStyle: {
          container: {
            bg: 'semantic.surface',
            borderColor: 'semantic.border',
            boxShadow: 'md', // Apply a standard shadow to cards
          },
        },
      },
      Box: {
        variants: {
          elevated: {
            bg: 'semantic.surface',
            boxShadow: 'md',
            borderRadius: 'md',
          },
          outlined: {
            bg: 'semantic.surface',
            borderWidth: '1px',
            borderColor: 'semantic.border',
            borderRadius: 'md',
          }
        }
      },
    },
    config: {
      initialColorMode: 'dark', // Setting dark as default to match your palette
      useSystemColorMode: false,
    }
  }
};

// Create and export the system
export const system = createSystem(defaultConfig, themeConfig);

// For backwards compatibility if needed
export default themeConfig.theme;