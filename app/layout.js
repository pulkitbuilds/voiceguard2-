import './globals.css';

export const metadata = {
  title: 'VoiceGuard — Real-Time Voice Clone & Spoof Detection',
  description:
    'AI-powered real-time detection and prevention of voice cloning impersonation attacks. Audio analysis, ML spoof classifier, and live risk-scoring dashboard.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg text-slate-100 antialiased">{children}</body>
    </html>
  );
}
