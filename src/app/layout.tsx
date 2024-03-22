import 'highlight.js/styles/github-dark.css'
import type {Metadata} from 'next'

import Root from '@/components/Root'

import './globals.css'

export const metadata: Metadata = {
  title: 'RAG APP',
  description: 'Simple RAG app integrated with TruLens',
}

export default function RootLayout({children}: {children: React.ReactNode}) {
  return <Root>{children}</Root>
}
