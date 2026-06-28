use crate::error::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(crate) struct SamplingConfig {
    pub do_sample: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
}

impl SamplingConfig {
    pub(crate) fn validate(self, name: &str) -> Result<()> {
        if self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err(Error::InvalidInput(format!(
                "{name}.top_p must be in (0, 1], got {}",
                self.top_p
            )));
        }
        if self.temperature <= 0.0 {
            return Err(Error::InvalidInput(format!(
                "{name}.temperature must be positive, got {}",
                self.temperature
            )));
        }
        Ok(())
    }
}

pub(crate) fn select_token(logits: &[f32], config: SamplingConfig, state: &mut u64) -> Result<i32> {
    if !config.do_sample {
        return argmax(logits);
    }
    sample(logits, config, state)
}

fn argmax(logits: &[f32]) -> Result<i32> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx as i32)
        .ok_or_else(|| Error::InvalidInput("cannot sample from empty logits".to_string()))
}

fn sample(logits: &[f32], config: SamplingConfig, state: &mut u64) -> Result<i32> {
    let mut candidates = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(Error::InvalidInput(
            "cannot sample from logits with no finite values".to_string(),
        ));
    }
    candidates.sort_by(|(_, a), (_, b)| b.total_cmp(a));
    if config.top_k > 0 && config.top_k < candidates.len() {
        candidates.truncate(config.top_k);
    }

    let max_logit = candidates[0].1;
    let mut probs = candidates
        .iter()
        .map(|(_, logit)| ((*logit - max_logit) / config.temperature).exp())
        .collect::<Vec<_>>();
    let sum = probs.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Ok(candidates[0].0 as i32);
    }
    for prob in &mut probs {
        *prob /= sum;
    }

    if config.top_p < 1.0 {
        let mut cumulative = 0.0f32;
        let mut keep = 0usize;
        for prob in &probs {
            keep += 1;
            cumulative += *prob;
            if cumulative >= config.top_p {
                break;
            }
        }
        candidates.truncate(keep.max(1));
        probs.truncate(candidates.len());
        let kept_sum = probs.iter().sum::<f32>();
        if kept_sum > 0.0 {
            for prob in &mut probs {
                *prob /= kept_sum;
            }
        }
    }

    let mut target = next_f32(state);
    for ((token, _), prob) in candidates.iter().zip(probs.iter()) {
        if target <= *prob {
            return Ok(*token as i32);
        }
        target -= *prob;
    }
    Ok(candidates
        .last()
        .map(|(token, _)| *token as i32)
        .unwrap_or(0))
}

fn next_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 40) as u32;
    (bits as f32) / ((1u32 << 24) as f32)
}
