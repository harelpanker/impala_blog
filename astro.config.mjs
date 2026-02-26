// @ts-check
import { defineConfig } from 'astro/config';

import mdx from '@astrojs/mdx';

// https://astro.build/config
export default defineConfig({
	// Update these for GitHub Pages:
	site: 'https://harelpanker.github.io',
	base: '/impala_blog',
	integrations: [mdx()],
});
