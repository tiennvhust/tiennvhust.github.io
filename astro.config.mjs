// @ts-check
import { defineConfig } from 'astro/config';

// https://astro.build/config
// This is a GitHub *user* site (tiennvhust.github.io), so it is served from
// the domain root — no `base` path needed.
export default defineConfig({
  site: 'https://tiennvhust.github.io',
});
