# Publication Figure Prompts for AI Image Generation

## Figure 1: System Architecture Diagram

### Prompt for Architecture Figure

```
Create a technical architecture diagram for a research paper showing a multi-agent robot navigation system with two main components arranged vertically:

**TOP SECTION - Decentralized GAT + TD3 Architecture:**

1. **Input Layer** (leftmost):
   - Robot state nodes: circles labeled "Robot i" containing [x, y, cos(θ), sin(θ), v_prev]
   - Obstacle state nodes: squares labeled "Obstacle j" containing [x, y, cos(θ), sin(θ)]
   - Use different colors: robots in blue circles, obstacles in gray squares

2. **Graph Attention Network (GAT) Encoder** (center-left):
   - Show heterogeneous graph structure with:
     * Robot nodes (blue circles) connected with bidirectional edges
     * Obstacle nodes (gray squares) with unidirectional edges TO robots only
   - Two parallel embedding MLPs:
     * Robot Encoder: Input(5) → Linear(128) → LeakyReLU → Linear(512) → LeakyReLU
     * Obstacle Encoder: Input(5) → Linear(128) → LeakyReLU → Linear(512) → LeakyReLU
   
3. **Dual Hard Attention Module** (center):
   - Two parallel branches:
     * **Robot-Robot Hard Attention:**
       - Input: [h_i, h_j, relative features (7-dim)]
       - MLP: Linear(512+7→512) → ReLU → Linear(512→512)
       - Binary classifier: Linear(512→2) → Gumbel-Softmax
       - Output: Binary mask A_rr
     
     * **Robot-Obstacle Hard Attention:**
       - Input: [h_i, h_o, relative features (5-dim)]
       - MLP: Linear(512+5→512) → ReLU → Linear(512→512)
       - Binary classifier: Linear(512→2) → Gumbel-Softmax
       - Output: Binary mask A_ro
   
4. **Goal-Oriented Soft Attention** (center-right):
   - MessagePassing layer with:
     * Query from robots: Q = Linear(512→512)
     * Key from edges: K = Linear(10→512)
     * Value from edges: V = Linear(10→512)
     * Attention scores: MLP([Q_i, K_ij], 1024→512→1) → Softmax
     * Aggregation: Σ(α_ij × V_ij)
   - Show attention weights as colored edges (warmer = stronger)

5. **TD3 Actor-Critic** (rightmost):
   - **Actor branch** (top):
     * Input: Concatenated [h_i, Σ messages] (1024-dim)
     * Decoder: Linear(1024→400) → LeakyReLU → Linear(400→300) → LeakyReLU → Linear(300→2) → Tanh
     * Output: [v_linear, ω_angular] for each robot
   
   - **Twin Critics** (bottom):
     * Critic 1: Linear(1024→400) → LeakyReLU → [Linear(400→300) + Linear(2→300)] → LeakyReLU → Linear(300→1)
     * Critic 2: Same architecture
     * Output: Q-values Q1(s,a), Q2(s,a)

**BOTTOM SECTION - Two-Tower Group Switcher (Supervised Ranking):**

1. **Input Features** (leftmost):
   - Show M candidate groups as stacked boxes
   - Each group contains:
     * Group embedding h_g (512-dim) - from pooling robot embeddings
     * Global embedding h_glob (512-dim) - mean of all robots
     * Scalar features (13-dim vector):
       - Base (5): size_feat, coupling_mode, A_in, A_out, A_obs
       - Extra group (5): mean_dist_goal, min_dist_goal, min_clearance, frac_reached, mean_heading_err
       - Extra global (3): var_dist_goal_global, frac_reached_global, steps_elapsed_frac

2. **Two-Tower Fusion Network** (center):
   - **Tower 1 - Embedding Tower:**
     * Input: [h_g || h_glob] (1024-dim)
     * Linear(1024→256) → GELU → LayerNorm(256)
     * Output: e' (256-dim)
   
   - **Tower 2 - Scalar Tower:**
     * Input: scalar features (13-dim)
     * Linear(13→64) → GELU → LayerNorm(64)
     * Output: s' (64-dim)
   
   - **Fusion Layer:**
     * Input: [e' || s'] (320-dim)
     * Linear(320→256) → GELU → LayerNorm(256) → Dropout(0.1)
     * Linear(256→1)
     * Output: logit score for each group

3. **Ranking Loss & Selection** (rightmost):
   - Show pairwise ranking with oracle scores
   - Loss: Pairwise Logistic Ranking Loss
   - Selection: argmax(logits) → Best group
   - Show sequential activation with coupling constraints:
     * ≤3 robots: rotation-free (parallel)
     * ≥4 robots: rotation-coupled (sequential)

**Visual Style:**
- Clean, academic diagram with clear data flow (left to right, top to bottom)
- Use consistent color scheme: blue for robots, gray for obstacles, green for attention, orange for actions
- Show tensor dimensions clearly at each layer
- Use arrows to indicate data flow
- Include mathematical notation where appropriate
- Add legend for symbols and colors
- Professional font (Arial or similar)
- White background, black text
- Include layer names and activation functions
```

---

## Figure 2: Attention Visualization Demonstration

### Prompt for Attention Visualization Figure

```
Create a technical demonstration figure showing multi-robot navigation with attention mechanisms in a top-down 2D environment:

**Environment Setup:**
- 2D gridded space (10m × 10m) with coordinate axes
- 14 blue circular robots (radius ~0.3m) at various positions
- 6-8 gray rectangular obstacles (static, various sizes 0.5m-1.5m)
- Goal positions shown as green stars/targets for each robot
- Include distance scale bar

**Main Visualization Panel (Large, center):**

1. **Robot States:**
   - Each robot drawn as blue circle with orientation arrow
   - Robot heading shown as black arrow from center
   - Goal direction shown as thin dashed line to green star
   - Label 2-3 robots as "Robot 1", "Robot 5", "Robot 14" for reference

2. **Obstacle Representation:**
   - Static obstacles as gray filled rectangles
   - Label as "Obs 1", "Obs 2", etc.
   - Show obstacle orientations with small markers

3. **Attention Weights Visualization:**
   - **Robot-Robot Attention (blue edges):**
     * Draw edges between robots with varying thickness/opacity
     * Edge thickness ∝ soft attention weight α_ij
     * Color gradient: light blue (weak) to dark blue (strong)
     * Show only edges with α_ij > 0.1 threshold for clarity
   
   - **Robot-Obstacle Attention (red edges):**
     * Draw edges from robots to nearby obstacles
     * Edge thickness ∝ attention weight α_io
     * Color gradient: light red (weak) to dark red (strong)
     * Show only edges with α_io > 0.1

4. **Hard Attention Masks (overlay):**
   - Edges with hard_weight = 1 shown as solid lines
   - Edges with hard_weight = 0 shown as dotted/absent
   - Optional: highlight hard edges with subtle glow effect

**Side Panels (Right side, stacked):**

1. **Attention Weight Matrix Heatmap (top):**
   - 14×14 heatmap for robot-robot attention
   - Color scale: white (0) to dark blue (1)
   - Diagonal blanked out (no self-attention)
   - Axes labeled "Source Robot" and "Target Robot"
   - Include colorbar with values

2. **Robot-Obstacle Attention Heatmap (middle):**
   - 14×8 heatmap (robots × obstacles)
   - Color scale: white (0) to dark red (1)
   - Axes labeled "Robot ID" and "Obstacle ID"
   - Include colorbar

3. **Feature Importance Bar Chart (bottom):**
   - Horizontal bars showing scalar features for selected group
   - Features: size_feat, coupling_mode, A_in, A_out, A_obs
   - Values normalized to [0,1]
   - Color-coded by feature type

**Inset Detail View (bottom-left corner):**
- Zoomed-in view of 3 robots forming a group
- Show detailed attention edges with numerical weights labeled
- Example: α_12 = 0.87, α_13 = 0.65
- Clearance distances to nearest obstacles with measurements
- Demonstrate coupling constraint activation

**Legend (top-right corner):**
- Robot (blue circle with arrow)
- Obstacle (gray rectangle)
- Goal (green star)
- Strong attention (thick edge, α > 0.5)
- Weak attention (thin edge, 0.1 < α < 0.5)
- Hard mask active (solid line)
- Hard mask inactive (dotted line)

**Annotations:**
- Add 2-3 callout boxes explaining:
  1. "High attention to nearby robot forming group"
  2. "Obstacle avoidance attention activated"
  3. "Low attention to distant robots (sparse graph)"

**Visual Style:**
- Publication quality, vector graphics style
- Clean white background
- High contrast for readability in print
- Consistent color palette matching Figure 1
- Grid lines subtle but visible
- Professional typography with clear labels
- Include scale indicators (1m reference bar)
```

---

## Figure 3 (Optional): Training Curves & Performance Comparison

### Prompt for Results Figure

```
Create a multi-panel figure showing training and evaluation results:

**Panel A - Training Curves (2×2 grid):**
1. Episode reward over time (GAT+TD3)
2. Success rate over episodes
3. Attention entropy over training
4. Switcher ranking loss convergence

**Panel B - Comparison Bar Charts:**
1. Success rate: Baseline vs GAT vs GAT+Switcher
2. Average steps to goal
3. Collision rate comparison
4. Computational efficiency

**Panel C - Ablation Study:**
- Table showing contribution of each component
- Metrics: Success %, Avg Reward, Collisions

**Style:** 
- Scientific publication standard (Nature/IEEE style)
- Error bars for statistical significance
- Consistent color scheme with architecture figures
- Clear axis labels with units
```

---

## Technical Specifications for All Figures

**Resolution & Format:**
- Minimum 300 DPI for publication
- Vector format preferred (SVG, EPS, or PDF)
- Fallback: High-res PNG (at least 3000px width)

**Color Palette (for consistency):**
- Robots: #2E86DE (blue)
- Obstacles: #95A5A6 (gray)
- Goals: #27AE60 (green)
- Robot-Robot Attention: #3498DB (light blue) to #0652DD (dark blue)
- Robot-Obstacle Attention: #FFC312 (yellow) to #EE5A6F (red)
- Actions: #F79F1F (orange)
- Neural network layers: #A29BFE (lavender)

**Typography:**
- Main labels: Arial/Helvetica, 10-12pt
- Annotations: Arial/Helvetica, 8-10pt
- Math symbols: Computer Modern or Times New Roman (italics)

**File Naming:**
- `fig1_architecture_diagram.pdf`
- `fig2_attention_visualization.pdf`
- `fig3_training_results.pdf`

---

## Notes for Image Generation Tools

**Recommended Tools:**
1. **For Architecture (Fig 1):** 
   - Use diagram-specific AI (e.g., DALL-E 3, Midjourney with technical diagram style)
   - Or create with draw.io / Lucidchart and ask AI to beautify
   - Consider using TikZ/LaTeX for publication quality

2. **For Attention Visualization (Fig 2):**
   - May need combination: matplotlib for heatmaps + AI for environment rendering
   - Can generate with your actual code data and enhance with AI styling

3. **Alternative Approach:**
   - Generate base figures with your simulation/training data
   - Use AI to enhance aesthetics and clarity
   - Ensure all labels and values are accurate to your system

**Checklist before submission:**
- [ ] All dimensions match your actual code
- [ ] Mathematical notation is correct
- [ ] Color scheme is colorblind-friendly
- [ ] Text is readable when printed in grayscale
- [ ] All acronyms defined in caption
- [ ] Figure referenced correctly in main text
