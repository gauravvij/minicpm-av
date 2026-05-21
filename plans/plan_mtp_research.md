# Plan: MTP for Vision-Language Models Research

## Goal
Investigate Multi-Token Prediction (MTP) behavior specifically for Vision-Language Models, using MiniCPM-V 4.6 as the testbed. This is a research contribution, not just enabling a flag.

## Research Summary
- **MTP Status**: Qwen3.5-0.8B natively supports MTP via `num_mtp_tokens` parameter
- **Research Gap**: No systematic study on how MTP performs with vision tokens in context
- **Architecture Constraint**: Qwen3.5 uses hybrid DeltaNet+Attention, blocking standard Medusa

## Research Questions

### RQ1: Vision Token Impact on MTP
- Does MTP prediction accuracy differ when vision tokens are present vs text-only?
- How does the position of vision tokens (prefix vs interleaved) affect MTP effectiveness?

### RQ2: Optimal MTP Configuration for 0.8B
- Is `num_mtp_tokens=2` optimal, or do different values (1, 3, 4) perform better?
- Trade-off between acceptance rate and computational overhead

### RQ3: Task-Specific MTP Behavior
- Does MTP help more for image captioning vs visual QA?
- Correlation between task complexity and MTP effectiveness

### RQ4: Edge Deployment Analysis
- Actual latency measurements on CPU/mobile hardware
- Memory overhead quantification
- Battery impact on mobile devices

## Subtasks

1. **Setup MTP Baseline**
   - Load MiniCPM-V 4.6 with MTP enabled
   - Verify basic functionality with vision inputs
   - Expected output: Working MTP inference script

2. **Vision Token Analysis (RQ1)**
   - Compare MTP acceptance rates: text-only vs image+text inputs
   - Vary vision token positions and measure impact
   - Expected output: RQ1 analysis report with metrics

3. **Optimal Configuration Study (RQ2)**
   - Sweep num_mtp_tokens from 1 to 4
   - Measure acceptance rate, perplexity, latency
   - Expected output: Optimal config recommendation

4. **Task-Specific Evaluation (RQ3)**
   - Benchmark on LLaVA-Bench (captioning) and MMBench (VQA)
   - Compare MTP vs non-MTP for each task type
   - Expected output: Task-specific recommendations

5. **Edge Deployment Profiling (RQ4)**
   - Measure latency on CPU (no GPU)
   - Memory usage profiling
   - Expected output: Deployment feasibility report

6. **Final Research Report**
   - Compile all findings
   - Provide recommendations for VLM-MTP best practices
   - Expected output: `mtp_vlm_research_report.md`

## Deliverables

| File | Description |
|------|-------------|
| `mtp_baseline.py` | MTP inference with MiniCPM-V |
| `rq1_vision_impact.py` | Vision token analysis script |
| `rq2_config_sweep.py` | Configuration optimization |
| `rq3_task_eval.py` | Task-specific evaluation |
| `rq4_edge_profile.py` | Edge profiling tools |
| `mtp_vlm_research_report.md` | Final research report |

## Evaluation Criteria
- [ ] RQ1: Quantified vision token impact on MTP acceptance rate
- [ ] RQ2: Optimal num_mtp_tokens identified with statistical significance
- [ ] RQ3: Task-specific MTP recommendations documented
- [ ] RQ4: Edge deployment metrics (latency, memory) measured
- [ ] Research report with actionable insights for VLM-MTP usage

## Notes
- **Priority**: Secondary to Audio Integration
- **Timeline**: Can run in parallel or after Audio Integration
- **Hardware**: Requires GPU for training, CPU for edge profiling
