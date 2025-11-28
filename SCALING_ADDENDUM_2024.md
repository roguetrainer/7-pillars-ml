# AI Scaling in 2024-2025: A Paradigm Shift
## Addendum to "The Seven Pillars of Statistical Wisdom in Modern AI/ML"

*Updated: November 2024*

---

## Executive Summary

**The "bigger is better" era of AI scaling may be ending.** After years of exponential growth in model size (GPT-3's 175B parameters → GPT-4's estimated 1.8T parameters), the industry is hitting fundamental limits with traditional pretraining scaling. However, a new paradigm is emerging: **test-time compute scaling** (inference-time reasoning), exemplified by OpenAI's o1 and o3 models.

**Key Developments:**
- Traditional pretraining scaling laws showing **diminishing returns**
- OpenAI's next model "Orion" reportedly **underperforming expectations**
- New focus on **test-time compute**: letting models "think" longer during inference
- o3 achieving breakthrough performance but at **$1,000-$10,000 per complex query**
- Industry shifting from "train bigger" to "think longer"

This represents a direct manifestation of **Pillar 2 (Information)**: the √N law is reasserting itself, forcing innovation in how we apply compute.

---

## The State of Traditional Scaling (Pretraining)

### What Was Working (2020-2023)

The "scaling hypothesis" drove the entire AI boom:
- **More parameters** + **more data** + **more compute** = better models
- Performance improved predictably along power-law curves
- About two-thirds of improvements in LLM performance over the last decade came from increases in model scale

**The Chinchilla Insight (2022):**
For compute-optimal training, model size and training tokens should scale equally: doubling the model size should be accompanied by doubling the number of training tokens

This revealed that most models (including GPT-3) were **undertrained**—they could have been better with more data rather than more parameters.

### What's Breaking Down (2024)

**Multiple sources now confirm traditional scaling is hitting limits:**

1. **Performance Plateaus:**
   - OpenAI's next-generation model Orion was one-quarter of the way through training, with performance on par with GPT-4's, despite larger architecture and more data
   - Next-generation large language models from OpenAI, Google, and Anthropic are falling short of expectations

2. **Industry Consensus:**
   - "If you just put in more compute, you put in more data, you make the model bigger — there are diminishing returns," said Anyscale co-founder Robert Nishihara
   - OpenAI and Safe Super Intelligence co-founder Ilya Sutskever told Reuters "everyone is looking for the next thing" to scale their AI models

3. **Data Scarcity:**
   - It's getting harder to find high-quality materials on the web that haven't already been tapped
   - Google and OpenAI have been disappointed with results of pretraining models on synthetic data

4. **Cost Explosion:**
   - The latest models cost as much as $100 million to train, and this could reach $100 billion within a few years

**This is the √N law in action:** To keep improving, you need exponentially more resources for diminishing returns.

---

## The New Paradigm: Test-Time Compute Scaling

### What Is Test-Time Compute?

Instead of making models bigger during training, we give them **more time and compute to "think" during inference** (when answering your question).

OpenAI's o1 model re-prompts itself several times over 10 to 30 seconds, breaking down a large problem into a series of smaller ones

**Key Insight:**
OpenAI's o1 model demonstrated that allowing more time to generate "thinking tokens" during inference correlated directly with improved accuracy across various tasks

### The Three Types of Scaling Laws (2024)

According to NVIDIA and industry researchers, we now have **three distinct scaling laws**:

1. **Pretraining Scaling** (traditional)
   - More data + more parameters during training
   - Status: Hitting diminishing returns

2. **Post-Training Scaling** (2023-2024)
   - Fine-tuning, RLHF, distillation, synthetic data
   - Status: Active area of development

3. **Test-Time Scaling / "Long Thinking"** (2024-2025)
   - More compute during inference
   - Status: **The new frontier**

### OpenAI o1 and o3: The Breakthrough

**o1 (September 2024):**
- First major "reasoning model"
- Uses chain-of-thought during inference
- Spends 10-30 seconds "thinking" before answering
- Dramatically outperforms GPT-4 on complex reasoning tasks
- o1 is trained via reinforcement learning to get better at implicit search via chain of thought

**o3 (December 2024):**
- Released just 3 months after o1
- **Record-breaking performance:**
  - 88% on ARC-AGI (where average human scores 75%)
  - 2727 on Codeforces (175th best competitive programmer on Earth)
  - 25% on FrontierMath (where each problem demands hours from expert mathematicians)

**The Cost:**
- o1: ~4× more expensive than GPT-4o per query
- o3 (high-compute mode): OpenAI used up to $10,000 worth of compute for each AGI answer, equivalent to 900 Nvidia H100 GPUs running for 8 hours
- OpenAI has reportedly weighed creating subscription plans costing up to $2,000

### How Much Does Test-Time Compute Help?

**The Equivalence:**
A 15× increase in inference-time compute would equate to a 10× increase in train-time compute

**In practical terms:**
- Letting a 200B parameter model "think" 4× longer can give performance equivalent to a 2.8T parameter model
- This is **dramatically more efficient** than actually training a 2.8T parameter model

**The Scaling Continues:**
o1's AIME accuracy increases at a constant rate with the logarithm of test-time compute

This means we're back to power-law scaling—just in a different dimension.

---

## Technical Deep Dive: How Test-Time Compute Works

### Chain-of-Thought (CoT)

The model generates hidden "thinking tokens" before giving its final answer. Instead of:

```
User: "What's 47 × 23?"
Model: "1,081"
```

It does:

```
User: "What's 47 × 23?"
Model [internal]: "Let me break this down. 47 × 20 = 940. 47 × 3 = 141. 
                   940 + 141 = 1,081. Let me verify: 23 × 40 = 920, 
                   23 × 7 = 161, 920 + 161 = 1,081. Confirmed."
Model [output]: "1,081"
```

### Reinforcement Learning for Better Reasoning

DeepSeek-R1 model exhibits sophisticated behaviors such as reflection, where it revisits and reevaluates its previous steps, and exploration of alternative problem-solving approaches

The model learns through RL to:
- Explore multiple solution paths
- Backtrack when stuck
- Verify its own reasoning
- Self-correct errors

### The "Aha Moment"

The model can figure out on its own that rethinking its approach leads to better answers—this fascinating "Aha moment" shows emergent self-improvement

---

## Implications and Trade-offs

### The Good News

1. **Scaling Continues:**
   Frontier training runs by 2030 are projected to be 5,000× larger than Llama 3.1 405B

2. **New Efficiency Frontier:**
   There are some problems we would be willing to spend millions of dollars to solve. A typical LLM query costs on the order of a penny. That's an easy eight orders of magnitude to explore

3. **Training Data Generation:**
   - One AI's inference time is a future AI's training time. OpenAI is using o1 to generate high-quality training data for "Orion," their next large language model

### The Challenges

1. **Cost Barriers:**
   - o3 is not AGI, and it still fails on some very easy tasks a human would do easily, despite costs of thousands of dollars per query
   - Only deep-pocketed institutions may afford high-compute versions

2. **Use Case Limitations:**
   - It doesn't seem like o3 would be anyone's "daily driver" like GPT-4o or Google Search might be. These models use too much compute for small questions
   - Better for "big picture prompts" like strategic planning than "How do I make pasta?"

3. **Environmental Concerns:**
   - Renowned AI researcher Yoshua Bengio noted that o1 "requires far more computing resources, and therefore energy"

4. **Latency Wall:**
   - Experiments indicate kernel latency on the order of 4.5 µs for A100 GPUs, which creates physical limits to parallelization
   - Can't just throw infinite GPUs at the problem

### The Skeptical View

**Is this just brute force?**

These brute force results typically looked quite linear during the regime where the system was climbing from 20% to 80%, meaning most progress was exponential in cost for each percentage point of gain

Philosopher Toby Ord argues that test-time compute might be less impressive than it appears:
- The logarithmic scaling means exponential costs for linear gains
- Similar to just running the model thousands of times and picking the best answer
- These o3 results came out months after o1, but could give a false impression of progress rate. Given historical rates, we're seeing where 5 years of progress should take us, not what 3 months delivered

---

## Connection to Pillar 2: The √N Law Reasserts Itself

This entire situation is a **textbook example of Pillar 2 (Information)**:

### Classical Statistics (1900s)
- Standard error ∝ 1/√N
- To halve uncertainty, need 4× more data
- Diminishing returns are fundamental

### Neural Scaling Laws (2020-2023)
- Loss ∝ (N_c / N)^α where α ≈ 0.05-0.10
- To halve loss, need roughly 10× more compute
- This worked brilliantly... until it didn't

### Test-Time Compute (2024-2025)
- Performance ∝ log(inference_compute)
- To gain another 10 percentage points, need 10× more thinking time
- **Same fundamental pattern, different axis**

**The Bitter Truth:**
You can't escape the √N law. You can only choose which dimension to scale:
- Training compute (hitting limits)
- Training data (running out)
- Inference compute (the new frontier)

But the diminishing returns are **always** there, because information accumulation is fundamentally sublinear.

---

## Industry Responses and Adaptations

### Major Labs' Strategies

**OpenAI:**
- Bet big on test-time compute (o1, o3)
- Using reasoning models to generate training data for next-gen base models
- Exploring $200-$2,000/month subscription tiers

**Google DeepMind:**
- Released Gemini 2.0 Flash Thinking (experimental reasoning model)
- Published research on optimal test-time compute scaling
- Combining approaches: pretraining + post-training + inference scaling

**Anthropic:**
- Anthropic co-founder Jack Clark predicted "2025 will see a combination of test-time scaling and traditional pre-training methods"
- Focusing on constitutional AI and safety during extended reasoning

**DeepSeek (China):**
- Released DeepSeek-R1 model with reinforcement learning for reasoning
- Achieved competitive performance with open weights
- Demonstrated that test-time compute is accessible beyond OpenAI

### Alternative Approaches

1. **Smaller, Better-Trained Models:**
   - Many developers already shifted to building smaller, more processing-efficient models, especially for edge devices

2. **Mixture of Experts (MoE):**
   - Mixtral 8×7B, GPT-4: activate only subset of parameters per query
   - More efficient than dense models at same capability level

3. **Synthetic Data:**
   - Still experimental, mixed results
   - Phi-1.5 showed promise with high-quality curated synthetic data

4. **Multimodal Scaling:**
   - Scaling across modalities (text, image, video, audio)
   - Different data distributions might extend runway

---

## What This Means for Practitioners

### For Researchers

1. **New Research Frontiers:**
   - How to make test-time compute more efficient
   - Better reward models for reinforcement learning
   - Closing the generator/verifier gap
   - Formal verification integration

2. **Theoretical Questions:**
   - Are there fundamental limits to test-time scaling?
   - What's the relationship between pretraining and inference compute?
   - Can we get better than logarithmic scaling?

3. **Practical Considerations:**
   - Most academic labs can't afford o3-level compute
   - Focus on efficient methods: distillation, pruning, quantization
   - Open-weight models (DeepSeek-R1, Llama) become more important

### For Practitioners Building Applications

1. **Cost-Benefit Analysis:**
   - Traditional models: Fast, cheap, good enough for most tasks
   - Reasoning models: Slow, expensive, necessary for complex problems
   - **Use the right tool for the right job**

2. **Architectural Implications:**
   - Design systems that can route queries appropriately
   - Simple questions → GPT-4o (~$0.01)
   - Complex reasoning → o1 (~$0.04-$0.20)
   - Critical decisions → o3 high-compute ($1,000+)

3. **Latency Considerations:**
   - Traditional: <1 second response
   - o1: 10-30 seconds
   - o3 high-compute: minutes to hours
   - **Plan UX accordingly**

### For Business Leaders

1. **Strategic Implications:**
   - AI capabilities will keep improving, just more expensively
   - The "AI gets cheaper and better" trend is bifurcating:
     - Commodity AI: cheaper, better, faster (GPT-4o, Gemini, Claude)
     - Expert AI: more expensive, much better, slower (o1, o3)

2. **Investment Decisions:**
   - Don't assume linear cost decreases
   - Budget for 10-100× cost increases for frontier capabilities
   - Consider which problems justify premium AI

3. **Competitive Dynamics:**
   - Deep pockets matter more (OpenAI raised $6.6B in October 2024)
   - Smaller players can compete with efficient methods and open weights
   - Application-level innovation remains accessible

---

## Future Predictions (2025-2030)

### Likely Developments

1. **Hybrid Scaling:**
   - Combination of test-time scaling and traditional pre-training methods to enhance model capabilities
   - Models that know when to think fast vs. slow

2. **Cost Reduction:**
   - Historical algorithmic improvements suggest 3× per year. o3's costs could drop by 1,000× in about 5 years, making it economical
   - Hardware improvements (H100 → B100 → beyond)

3. **Specialized Reasoning:**
   - Domain-specific reasoning models (medicine, law, mathematics)
   - Formal verification for critical applications

4. **Agentic Systems:**
   - Models that can run for hours/days on complex tasks
   - "OpenAI's o1 thinks for seconds, but we aim for future versions to think for hours, days, even weeks," said Dr. Noam Brown

### Open Questions

1. **Does test-time scaling have limits?**
   - Will it hit diminishing returns like pretraining did?
   - What's the theoretical ceiling?

2. **Can we do better than logarithmic?**
   - Current scaling: performance ∝ log(compute)
   - Are there architectural breakthroughs that could improve this?

3. **What about AGI timeline?**
   - o3's 88% on ARC-AGI is impressive but not AGI
   - o3 still fails on some very easy tasks a human would do easily
   - Scaling alone may not be sufficient

---

## Conclusion: Pillar 2 Never Sleeps

The 2024-2025 AI scaling crisis perfectly validates Stigler's Second Pillar:

**Information accumulates with √N, not N.**

We cannot escape this fundamental law. We can only:
1. Choose which dimension to scale (training → inference)
2. Find more efficient methods (MoE, distillation)
3. Accept that exponential improvement requires exponential resources

The industry's pivot from "train bigger" to "think longer" isn't abandoning scaling—it's **respecting** the information laws that Stigler identified. The diminishing returns were always coming. Physics and mathematics don't care about hype cycles.

**The silver lining:** Test-time compute gives us a new axis to scale along, potentially buying another 5-10 years of improvement before we hit limits there too. And who knows—maybe by then we'll have discovered yet another dimension.

But Pillar 2 will still be there, patiently reminding us: **there's no such thing as a free lunch in information theory.**

---

## References

### Key Papers & Blog Posts

1. OpenAI (2024). "Learning to Reason with LLMs" (o1 announcement)
2. Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
3. Kaplan et al. (2020). "Scaling Laws for Neural Language Models" (OpenAI)
4. Google DeepMind (2024). "Scaling LLM Test-Time Compute Optimally"
5. DeepSeek (2024). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs"

### Industry Analysis

- TechCrunch: "Current AI scaling laws showing diminishing returns" (Nov 2024)
- The Batch (deeplearning.ai): "AI Giants Rethink Model Training Strategy" (Nov 2024)
- Foundation Capital: "Has AI scaling hit a limit?" (Dec 2024)
- Epoch AI: "Can AI scaling continue through 2030?" (Aug 2024)

### Technical Discussions

- Toby Ord: "Inference Scaling and the Log-x Chart"
- LessWrong: "o1: A Technical Primer"
- LessWrong: "Implications of the inference scaling paradigm for AI safety"
- Hugging Face: "What is test-time compute and how to scale it?"

---

*This addendum will be updated as the situation evolves. Last update: November 2024*

**Recommended for:** Updating the main notebook with current events, understanding industry trends, and contextualizing Pillar 2 with breaking news.
