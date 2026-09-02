/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,jsx}',
    './components/**/*.{js,jsx}',
  ],
  theme: {
    extend: {
      colors: {
        bg: '#0b0f14',
        panel: '#121821',
        panel2: '#161d28',
        line: '#232c3a',
        accent: '#5eead4',
        danger: '#f87171',
        warn: '#fbbf24',
        safe: '#34d399',
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
    },
  },
  plugins: [],
};
