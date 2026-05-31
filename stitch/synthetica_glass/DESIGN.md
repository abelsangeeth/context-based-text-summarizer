# Design System Specification: Vision-Augmented LLM Pipeline for Long-Form Document Summarization

## 1. Overview & Creative North Star: "The Architectural Curator"
This design system moves beyond the "SaaS template" aesthetic to establish a visual language of high-trust, research-grade precision. Our Creative North Star is **The Architectural Curator**. 

The UI should feel like a high-end gallery or a modern laboratory: quiet, expansive, and meticulously organized. We break the standard "box-in-a-box" layout by using intentional asymmetry, generous white space, and a shift from structural lines to **tonal layering**. The goal is to make the AI’s summarization feel like a premium, human-curated insight rather than a machine-generated data dump.

---

## 2. Colors & Tonal Depth
We leverage the Material Design convention to create a sophisticated, low-friction environment. 

### The "No-Line" Rule
**Explicit Instruction:** Do not use 1px solid borders to section off content. Boundaries must be defined solely through background color shifts. For example, a `surface-container-low` section sitting on a `surface` background provides enough contrast to denote a container without the visual "noise" of a stroke.

### Surface Hierarchy & Nesting
Treat the UI as a series of physical layers. Use the surface-container tiers to define importance:
- **Base Layer:** `surface` (#f9f9ff)
- **Secondary Content:** `surface-container-low` (#f2f3fd)
- **Interactive Cards/Containers:** `surface-container-highest` (#e1e2ec)
- **Floating Overlays:** `surface-container-lowest` (#ffffff) with backdrop blur.

### Signature Textures & Glassmorphism
- **The AI Insight Glow:** For primary summaries or "Vision-Augmented LLM Pipeline for Long-Form Document Summarization" highlights, use a subtle linear gradient: `primary` (#0058be) to `primary-container` (#2170e4) at a 135-degree angle.
- **Glassmorphism:** Floating navigation bars or modal overlays must use semi-transparent `surface_container_lowest` (80% opacity) with a `12px` backdrop-blur. This softens the edges and makes the platform feel "airy" and integrated.

---

## 3. Typography: Editorial Authority
We use a dual-typeface system to balance technical precision with modern sophistication.

*   **Display & Headlines (Manrope):** High-impact, geometric, and authoritative. Use `display-lg` (3.5rem) for landing hero sections to create a sense of scale.
*   **Body & Labels (Inter):** Optimized for high-readability in dense research summaries.
*   **The "Context" Shift:** Use `label-sm` in all-caps with `0.05em` letter-spacing for category tags or metadata to provide a "research-journal" feel.

| Role | Font | Size | Weight | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Display LG** | Manrope | 3.5rem | 700 | Hero Statements |
| **Headline MD** | Manrope | 1.75rem | 600 | Section Headers |
| **Title LG** | Inter | 1.375rem | 500 | Card Titles / Summary Headers |
| **Body LG** | Inter | 1rem | 400 | Main Summary Text |
| **Label MD** | Inter | 0.75rem | 600 | Metadata / Buttons |

---

## 4. Elevation & Depth
In this design system, depth is a functional tool, not a decorative one.

### The Layering Principle
Achieve hierarchy by "stacking" surface tokens. Place a `surface-container-lowest` card on a `surface-container-low` background. This creates a soft, natural lift that communicates "interactivity" without needing a shadow.

### Ambient Shadows
When a floating effect is required (e.g., a "Summarize" FAB or a dropdown), use **Ambient Shadows**:
- **Shadow Color:** Tinted with `on-surface` (#191b23) at 6% opacity.
- **Blur:** Large (24px to 40px) with 0 offset. This mimics natural light rather than a harsh digital drop shadow.

### The "Ghost Border" Fallback
If contrast is required for accessibility, use a **Ghost Border**: The `outline-variant` (#c2c6d6) at 20% opacity. **Never use 100% opaque borders.**

---

## 5. Components & Primitives

### Buttons (The "Call to Action")
- **Primary:** Gradient from `primary` to `primary_container`. `0.5rem` (lg) corner radius. No border.
- **Secondary:** `surface_container_high` background with `on_secondary_container` text.
- **Tertiary:** No background. `primary` text. Use `label-md` for weight.

### Cards & Summarization Blocks
- **Forbid dividers.** Use `spacing-8` (2rem) of vertical white space to separate summary sections.
- **Nesting:** Place AI-generated text inside a `surface_container_low` block to distinguish it from user-inputted text.

### Inputs & Text Areas
- **State:** Unfocused inputs should use `surface_variant` (#e1e2ec) with no border. 
- **Focus:** Transition to a `ghost border` (primary color at 30% opacity) and a subtle inner glow.

### Signature Component: The "Context-Breadcrumb"
For research-grade navigation, use a horizontal list of `Chips` using `surface_container_highest`. These chips should have `0.25rem` (md) radius and use `label-sm` for technical metadata (e.g., "Source: PDF", "Confidence: 98%").

---

## 6. Do's and Don'ts

### Do:
- **Embrace Asymmetry:** Align the summary text to the left but place metadata or "key takeaways" in a floating right-hand column using a different surface tone.
- **Use "Breathing Room":** Default to `spacing-10` or `spacing-12` for section margins. Precision is born from clarity.
- **Respect the Tones:** Use `on_surface_variant` for secondary text to maintain a soft, high-end visual gray-scale.

### Don't:
- **Don't use "Pure Black":** Use `on_surface` (#191b23) for text. It’s easier on the eyes for long-form reading.
- **Don't use Dividers:** Avoid horizontal rules (`<hr>`). Use background shifts or white space to denote the end of a thought.
- **Don't Over-Round:** Stick to `0.5rem` (lg) for most components. `full` (pill-shape) is reserved only for status chips or small buttons to avoid looking "childish."