# tiennvhust.github.io

Personal portfolio site, built with [Astro](https://astro.build) + TypeScript and
deployed to GitHub Pages.

## ✏️ Updating content (the important part)

**Almost all content lives in one file:** [`src/data/resume.ts`](src/data/resume.ts).

It's a typed file, so your editor will autocomplete fields and flag mistakes.
To update the site you usually just edit a list:

| To change… | Edit this export in `src/data/resume.ts` |
| --- | --- |
| Name / tagline / CV PDF / chatbot URL | `profile` |
| Email, LinkedIn, GitHub links | `contacts` |
| Degrees | `education` |
| Jobs | `experience` |
| Projects (with images/videos) | `projects` |
| Papers | `publications` |

Adding a new job, for example, is just adding one object to the `experience` array —
no HTML editing required. Images/videos/the PDF live in [`public/`](public/) and are
referenced with a leading slash, e.g. `/images/design.png`.

## 🧩 Structure

```
src/
  data/resume.ts        <- YOUR CONTENT
  pages/index.astro     <- assembles the sections
  layouts/Base.astro    <- <head>, fonts, analytics, scroll-reveal
  components/           <- Nav, Hero, Timeline, Projects, Publications, Contact, ChatWidget
  styles/global.css     <- design tokens (colors, spacing) + base styles
public/                 <- static assets (files, images, videos, favicon)
```

To restyle globally, change the CSS variables at the top of
[`src/styles/global.css`](src/styles/global.css) (e.g. `--maroon`, `--bg`).

## 🛠️ Local development

```bash
npm install      # first time only
npm run dev      # local dev server with hot reload (http://localhost:4321)
npm run build    # production build into dist/
npm run preview  # preview the production build
npm run check    # TypeScript / Astro type checking
```

## 🚀 Deployment

Pushing to `main` triggers [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml),
which builds the site and publishes `dist/` to GitHub Pages automatically.

> Note: GitHub Pages must be set to **Source: GitHub Actions** in the repo settings
> (Settings → Pages). It already was for the previous setup.
