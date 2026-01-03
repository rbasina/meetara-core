import type { Metadata } from 'next'
import { Plus_Jakarta_Sans, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const jakarta = Plus_Jakarta_Sans({ 
  subsets: ['latin'],
  variable: '--font-sans',
  display: 'swap'
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap'
})

export const metadata: Metadata = {
  title: 'me²TARA - Knowledge-Based AI Assistant',
  description: 'Intelligent AI assistant with domain-specific knowledge retrieval and expert document analysis.',
  keywords: ['AI', 'RAG', 'Knowledge Assistant', 'LLM', 'Document Analysis'],
  authors: [{ name: 'Meetara Lab' }],
  icons: {
    icon: '/favicon.ico',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `
              try {
                const theme = localStorage.getItem('meetara-theme') || 'system';
                const systemDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
                const resolvedTheme = theme === 'system' ? (systemDark ? 'dark' : 'light') : theme;
                document.documentElement.classList.add(resolvedTheme);
              } catch (e) {}
            `,
          }}
        />
      </head>
      <body className={`${jakarta.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        {children}
      </body>
    </html>
  )
}
